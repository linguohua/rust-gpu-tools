//! The CUDA specific implementation of a [`Buffer`], [`Device`], [`Program`] and [`Kernel`].

pub(crate) mod utils;

use std::ffi::{c_void, CStr, CString};
use std::hash::{Hash, Hasher};

use rustacuda::memory::AsyncCopyDestination;

use crate::device::{DeviceUuid, PciId, Vendor};
use crate::error::{GPUError, GPUResult};

/// A Buffer to be used for sending and receiving data to/from the GPU.
pub struct Buffer<T> {
    // We cannot use `T` directly for the `DeviceBuffer` as `AsyncCopyDestination` is only
    // implemented for `u8`.
    buffer: rustacuda::memory::DeviceBuffer<u8>,
    /// The number of T-sized elements.
    length: usize,
    _phantom: std::marker::PhantomData<T>,
}

/// CUDA specific device.
#[derive(Debug, Clone)]
pub struct Device {
    vendor: Vendor,
    name: String,
    memory: u64,
    pci_id: PciId,
    uuid: Option<DeviceUuid>,
    device: rustacuda::device::Device,
    context: rustacuda::context::UnownedContext,
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
}

/// Abstraction that contains everything to run a CUDA kernel on a GPU.
///
/// The majority of methods are the same as [`crate::opencl::Program`], so you can write code using this
/// API, which will then work with OpenCL as well as CUDA kernels.
#[allow(broken_intra_doc_links)]
pub struct Program {
    context: rustacuda::context::UnownedContext,
    module: rustacuda::module::Module,
    stream: rustacuda::stream::Stream,
    device_name: String,
}

impl Program {
    /// Returns the name of the GPU, e.g. "GeForce RTX 3090".
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Creates a program for a specific device from a compiled CUDA binary.
    pub fn from_binary(device: &Device, filename: &CStr) -> GPUResult<Program> {
        rustacuda::context::CurrentContext::set_current(&device.context)?;
        let module = rustacuda::module::Module::load_from_file(filename)?;
        let stream =
            rustacuda::stream::Stream::new(rustacuda::stream::StreamFlags::NON_BLOCKING, None)?;
        let prog = Program {
            module,
            stream,
            device_name: device.name(),
            context: device.context.clone(),
        };
        rustacuda::context::ContextStack::pop().expect("Cannot remove newly created context.");
        Ok(prog)
    }

    /// Creates a new buffer that can be used for input/output with the GPU.
    ///
    /// The `length` is the number of elements to create.
    pub fn create_buffer<T>(&self, length: usize) -> GPUResult<Buffer<T>> {
        assert!(length > 0);
        let buffer = unsafe {
            rustacuda::memory::DeviceBuffer::<u8>::uninitialized(length * std::mem::size_of::<T>())?
        };

        Ok(Buffer::<T> {
            buffer,
            length,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Returns a kernel.
    ///
    /// The `global_work_size` does *not* follow the OpenCL definition. It is *not* the total
    /// number of threads. Instead it follows CUDA's definition and is the number of
    /// `local_work_size` sized thread groups. So the total number of threads is
    /// `global_work_size * local_work_size`.
    pub fn create_kernel(&self, name: &str, gws: usize, lws: usize) -> GPUResult<Kernel> {
        let function_name = CString::new(name).expect("Kernel name must not contain nul bytes");
        let function = self.module.get_function(&function_name)?;

        Ok(Kernel {
            function,
            global_work_size: gws,
            local_work_size: lws,
            stream: &self.stream,
            args: Vec::new(),
        })
    }

    /// Puts data from an existing buffer onto the GPU.
    ///
    /// The `offset` is in number of `T` sized elements, not in their byte size.
    pub fn write_from_buffer<T>(
        &self,
        buffer: &mut Buffer<T>,
        offset: usize,
        data: &[T],
    ) -> GPUResult<()> {
        assert!(offset + data.len() <= buffer.length);
        unsafe {
            let bytes = std::slice::from_raw_parts(
                data.as_ptr() as *const T as *const u8,
                data.len() * std::mem::size_of::<T>(),
            );
            buffer.buffer.async_copy_from(bytes, &self.stream)?;
        };
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
        assert!(offset + data.len() <= buffer.length);

        unsafe {
            let bytes = std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut T as *mut u8,
                data.len() * std::mem::size_of::<T>(),
            );
            buffer.buffer.async_copy_to(bytes, &self.stream)?;
        };

        Ok(())
    }

    /// Run some code in the context of the program
    ///
    /// It sets the correct contexts and synchronizes the stream before returning.
    ///
    /// It takes the program as a parameter, so that we can use the same function body, for both
    /// the OpenCL and the CUDA code path. The only difference is the type of the program.
    pub fn run<F, R, E>(&self, fun: F) -> Result<R, E>
    where
        F: FnOnce(&Self) -> Result<R, E>,
        E: From<GPUError>,
    {
        rustacuda::context::CurrentContext::set_current(&self.context).map_err(Into::into)?;
        let result = fun(self);
        self.stream.synchronize().map_err(Into::into)?;
        rustacuda::context::ContextStack::pop().map_err(Into::into)?;
        result
    }
}

// TODO vmx 2021-07-07: Check if RustaCUDA types use in `Program` can be made `Send`, so that
// this manual `Send` implementation is no longer needed.
unsafe impl Send for Program {}

/// Abstraction for kernel arguments.
///
/// Kernel arguments implement this trait, so that they can bed convert it into the correct
/// pointers needed by the actual kernel call.
pub trait KernelArgument<'a> {
    /// Converts into a C void pointer.
    fn as_c_void(&self) -> *mut c_void;
}

impl<'a, T> KernelArgument<'a> for Buffer<T> {
    fn as_c_void(&self) -> *mut c_void {
        &self.buffer as *const _ as _
    }
}

impl KernelArgument<'_> for i32 {
    fn as_c_void(&self) -> *mut c_void {
        self as *const _ as _
    }
}
impl KernelArgument<'_> for u32 {
    fn as_c_void(&self) -> *mut c_void {
        self as *const _ as _
    }
}

/// A kernel that can be executed.
#[derive(Debug)]
pub struct Kernel<'a> {
    function: rustacuda::function::Function<'a>,
    global_work_size: usize,
    local_work_size: usize,
    stream: &'a rustacuda::stream::Stream,
    args: Vec<*mut c_void>,
}

impl<'a> Kernel<'a> {
    /// Add an argument to the kernel. All arguments are positional, so you need to make sure you
    /// call them in the expected order.
    pub fn arg<T: KernelArgument<'a>>(mut self, t: &T) -> Self {
        self.args.push(t.as_c_void());
        self
    }

    /// Actually run the kernel.
    pub fn run(self) -> GPUResult<()> {
        unsafe {
            self.stream.launch(
                &self.function,
                self.global_work_size as u32,
                self.local_work_size as u32,
                0,
                &self.args,
            )?;
        };
        Ok(())
    }
}
