//! The OpenCL specific implementation of a [`Buffer`], [`Device`], [`Program`] and [`Kernel`].

pub(crate) mod utils;

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ptr;

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::error_codes::ClError;
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::CL_MEM_READ_WRITE;
use opencl3::program::ProgramInfo::CL_PROGRAM_BINARIES;
use opencl3::types::CL_BLOCKING;

use crate::device::{DeviceUuid, PciId, Vendor};
use crate::error::{GPUError, GPUResult};

/// The lowest level identifier of an OpenCL device, it changes whenever a device is initialized.
#[allow(non_camel_case_types)]
pub type cl_device_id = opencl3::types::cl_device_id;

/// A Buffer to be used for sending and receiving data to/from the GPU.
pub struct Buffer<T> {
    buffer: opencl3::memory::Buffer<T>,
    /// The number of T-sized elements.
    length: usize,
}

/// OpenCL specific device.
#[derive(Debug, Clone)]
pub struct Device {
    vendor: Vendor,
    name: String,
    memory: u64,
    pci_id: PciId,
    uuid: Option<DeviceUuid>,
    device: opencl3::device::Device,
}

impl Hash for Device {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pci_id.hash(state);
        self.uuid.hash(state);
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        self.pci_id == other.pci_id && self.uuid == other.uuid
    }
}

impl Eq for Device {}

impl Device {
    /// Returns the [`Vendor`] of the GPU.
    pub fn vendor(&self) -> Vendor {
        self.vendor
    }

    /// Returns the name of the GPU, e.g. "GeForce RTX 3090".
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Returns the memory of the GPU in bytes.
    pub fn memory(&self) -> u64 {
        self.memory
    }

    /// Returns the PCI-ID of the GPU, see the [`PciId`] type for more information.
    pub fn pci_id(&self) -> PciId {
        self.pci_id
    }

    /// Returns the PCI-ID of the GPU if available, see the [`DeviceUuid`] type for more
    /// information.
    pub fn uuid(&self) -> Option<DeviceUuid> {
        self.uuid
    }

    /// Low-level access to the device identifier.
    ///
    /// It changes when the device is initialized and should only be used to interact with other
    /// libraries that work on the lowest OpenCL level.
    pub fn cl_device_id(&self) -> cl_device_id {
        self.device.id()
    }
}

/// Abstraction that contains everything to run an OpenCL kernel on a GPU.
///
/// The majority of methods are the same as [`crate::cuda::Program`], so you can write code using this
/// API, which will then work with OpenCL as well as CUDA kernels.
#[allow(broken_intra_doc_links)]
pub struct Program {
    device_name: String,
    queue: CommandQueue,
    context: Context,
    kernels_by_name: HashMap<String, opencl3::kernel::Kernel>,
}

impl Program {
    /// Returns the name of the GPU, e.g. "GeForce RTX 3090".
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Creates a program for a specific device from OpenCL source code.
    pub fn from_opencl(device: &Device, src: &str) -> GPUResult<Program> {
        let cached = utils::cache_path(device, src)?;
        if std::path::Path::exists(&cached) {
            let bin = std::fs::read(cached)?;
            Program::from_binary(device, bin)
        } else {
            let context = Context::from_device(&device.device)?;
            let mut program = opencl3::program::Program::create_from_source(&context, src)?;
            if let Err(build_error) = program.build(context.devices(), "") {
                let log = program.get_build_log(context.devices()[0])?;
                return Err(GPUError::Opencl3(build_error, Some(log)));
            }
            let queue = CommandQueue::create(&context, context.default_device(), 0)?;
            let kernels = opencl3::kernel::create_program_kernels(&program)?;
            let kernels_by_name = kernels
                .into_iter()
                .map(|kernel| {
                    let name = kernel.function_name()?;
                    Ok((name, kernel))
                })
                .collect::<Result<_, ClError>>()?;
            let prog = Program {
                device_name: device.name(),
                queue,
                context,
                kernels_by_name,
            };
            let binaries = program
                .get_binaries()
                .map_err(|_| GPUError::ProgramInfoNotAvailable(CL_PROGRAM_BINARIES))?;
            std::fs::write(cached, binaries[0].clone())?;
            Ok(prog)
        }
    }

    /// Creates a program for a specific device from a compiled OpenCL binary.
    pub fn from_binary(device: &Device, bin: Vec<u8>) -> GPUResult<Program> {
        let context = Context::from_device(&device.device)?;
        let bins = vec![&bin[..]];
        let mut program =
            opencl3::program::Program::create_from_binary(&context, context.devices(), &bins)?;
        if let Err(build_error) = program.build(context.devices(), "") {
            let log = program.get_build_log(context.devices()[0])?;
            return Err(GPUError::Opencl3(build_error, Some(log)));
        }
        let queue = CommandQueue::create(&context, context.default_device(), 0)?;
        let kernels = opencl3::kernel::create_program_kernels(&program)?;
        let kernels_by_name = kernels
            .into_iter()
            .map(|kernel| {
                let name = kernel.function_name()?;
                Ok((name, kernel))
            })
            .collect::<Result<_, ClError>>()?;
        Ok(Program {
            device_name: device.name(),
            queue,
            context,
            kernels_by_name,
        })
    }

    /// Creates a new buffer that can be used for input/output with the GPU.
    ///
    /// The `length` is the number of elements to create.
    pub fn create_buffer<T>(&self, length: usize) -> GPUResult<Buffer<T>> {
        assert!(length > 0);
        let buff = opencl3::memory::Buffer::create(
            &self.context,
            CL_MEM_READ_WRITE,
            length,
            ptr::null_mut(),
        )?;

        Ok(Buffer::<T> {
            buffer: buff,
            length,
        })
    }

    /// Returns a kernel.
    ///
    /// The `global_work_size` does *not* follow the OpenCL definition. It is *not* the total
    /// number of threads. Instead it follows CUDA's definition and is the number of
    /// `local_work_size` sized thread groups. So the total number of threads is
    /// `global_work_size * local_work_size`.
    pub fn create_kernel(
        &self,
        name: &str,
        global_work_size: usize,
        local_work_size: usize,
    ) -> GPUResult<Kernel> {
        let kernel = self
            .kernels_by_name
            .get(name)
            .ok_or_else(|| GPUError::KernelNotFound(name.to_string()))?;
        let mut builder = ExecuteKernel::new(&kernel);
        builder.set_global_work_size(global_work_size * local_work_size);
        builder.set_local_work_size(local_work_size);
        Ok(Kernel {
            builder,
            queue: &self.queue,
        })
    }

    /// Puts data from an existing buffer onto the GPU.
    ///
    /// The `offset` is in number of `T` sized elements, not in their byte size.
    pub fn write_from_buffer<T>(
        &self,
        buffer: &Buffer<T>,
        offset: usize,
        data: &[T],
    ) -> GPUResult<()> {
        assert!(offset + data.len() <= buffer.length, "Buffer is too small");

        let buff = buffer
            .buffer
            .create_sub_buffer(CL_MEM_READ_WRITE, offset, data.len())?;

        self.queue
            .enqueue_write_buffer(&buff, CL_BLOCKING, 0, data, &[])?;

        Ok(())
    }

    /// Reads data from the GPU into an existing buffer.
    ///
    /// The `offset` is in number of `T` sized elements, not in their byte size.
    pub fn read_into_buffer<T>(
        &self,
        buffer: &Buffer<T>,
        offset: usize,
        data: &mut [T],
    ) -> GPUResult<()> {
        assert!(offset + data.len() <= buffer.length, "Buffer is too small");
        let buff = buffer
            .buffer
            .create_sub_buffer(CL_MEM_READ_WRITE, offset, data.len())?;

        self.queue
            .enqueue_read_buffer(&buff, CL_BLOCKING, 0, data, &[])?;

        Ok(())
    }

    /// Run some code in the context of the program
    ///
    /// It takes the program as a parameter, so that we can use the same function body, for both
    /// the OpenCL and the CUDA code path. The only difference is the type of the program.
    pub fn run<F, R, E>(&self, fun: F) -> Result<R, E>
    where
        F: FnOnce(&Self) -> Result<R, E>,
        E: From<GPUError>,
    {
        fun(self)
    }
}

/// Abstraction for kernel arguments.
///
/// The kernel doesn't support being called with custom types, hence some conversion might be
/// needed. This trait enables automatic coversions, so that any type implementing it can be
/// passed into a [`Kernel`].
pub trait KernelArgument<'a> {
    /// Apply the kernel argument to the kernel.
    fn push(&self, kernel: &mut Kernel<'a>);
}

impl<'a, T> KernelArgument<'a> for Buffer<T> {
    fn push(&self, kernel: &mut Kernel<'a>) {
        kernel.builder.set_arg(&self.buffer);
    }
}

impl KernelArgument<'_> for i32 {
    fn push(&self, kernel: &mut Kernel) {
        kernel.builder.set_arg(self);
    }
}

impl KernelArgument<'_> for u32 {
    fn push(&self, kernel: &mut Kernel) {
        kernel.builder.set_arg(self);
    }
}

/// A local buffer.
pub struct LocalBuffer<T> {
    /// The number of T sized elements.
    length: usize,
    _phantom: std::marker::PhantomData<T>,
}
impl<T> LocalBuffer<T> {
    /// Returns a new buffer of the specified `length`.
    pub fn new(length: usize) -> Self {
        LocalBuffer::<T> {
            length,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> KernelArgument<'_> for LocalBuffer<T> {
    fn push(&self, kernel: &mut Kernel) {
        kernel
            .builder
            .set_arg_local_buffer::<T>(self.length * std::mem::size_of::<T>());
    }
}

/// A kernel that can be executed.
#[derive(Debug)]
pub struct Kernel<'a> {
    builder: ExecuteKernel<'a>,
    queue: &'a CommandQueue,
}

impl<'a> Kernel<'a> {
    /// Add an argument to the kernel. All arguments are positional, so you need to make sure you
    /// call them in the expected order.
    pub fn arg<T: KernelArgument<'a>>(mut self, t: &T) -> Self {
        t.push(&mut self);
        self
    }

    /// Actually run the kernel.
    pub fn run(mut self) -> GPUResult<()> {
        self.builder.enqueue_nd_range(&self.queue)?;
        Ok(())
    }
}
