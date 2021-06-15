mod error;
mod utils;

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ptr;

pub use error::{GPUError, GPUResult};

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::DeviceInfo::CL_DEVICE_ENDIAN_LITTLE;
use opencl3::error_codes::ClError;
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::CL_MEM_READ_WRITE;
use opencl3::program::ProgramInfo::CL_PROGRAM_BINARIES;
use opencl3::types::CL_BLOCKING;

pub type BusId = u32;

#[allow(non_camel_case_types)]
pub type cl_device_id = opencl3::types::cl_device_id;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Brand {
    Amd,
    Apple,
    Nvidia,
}

impl Brand {
    /// Returns a brand by name if it exists
    fn by_name(name: &str) -> Option<Self> {
        match name {
            "NVIDIA CUDA" => Some(Self::Nvidia),
            "AMD Accelerated Parallel Processing" => Some(Self::Amd),
            "Apple" => Some(Self::Apple),
            _ => None,
        }
    }
}

impl fmt::Display for Brand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let brand = match self {
            Brand::Nvidia => "NVIDIA CUDA",
            Brand::Amd => "AMD Accelerated Parallel Processing",
            Brand::Apple => "Apple",
        };
        write!(f, "{}", brand)
    }
}

pub struct Buffer<T> {
    buffer: opencl3::memory::Buffer<T>,
    length: usize,
}

#[derive(Debug, Clone)]
pub struct Device {
    brand: Brand,
    name: String,
    memory: u64,
    bus_id: Option<BusId>,
    device: opencl3::device::Device,
}

impl Hash for Device {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bus_id.hash(state);
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        self.bus_id == other.bus_id
    }
}

impl Eq for Device {}

impl Device {
    pub fn brand(&self) -> Brand {
        self.brand
    }
    pub fn name(&self) -> String {
        self.name.clone()
    }
    pub fn memory(&self) -> u64 {
        self.memory
    }
    pub fn is_little_endian(&self) -> GPUResult<bool> {
        match self.device.endian_little() {
            Ok(0) => Ok(false),
            Ok(_) => Ok(true),
            Err(_) => Err(GPUError::DeviceInfoNotAvailable(CL_DEVICE_ENDIAN_LITTLE)),
        }
    }
    pub fn bus_id(&self) -> Option<BusId> {
        self.bus_id
    }

    /// Return all available GPU devices of supported brands.
    pub fn all() -> Vec<&'static Device> {
        Self::all_iter().collect()
    }

    pub fn by_bus_id(bus_id: BusId) -> GPUResult<&'static Device> {
        Self::all_iter()
            .find(|d| match d.bus_id {
                Some(id) => bus_id == id,
                None => false,
            })
            .ok_or(GPUError::DeviceNotFound)
    }

    pub fn cl_device_id(&self) -> cl_device_id {
        self.device.id()
    }

    fn all_iter() -> impl Iterator<Item = &'static Device> {
        utils::DEVICES.iter()
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy)]
pub enum GPUSelector {
    BusId(u32),
    Index(usize),
}

impl GPUSelector {
    pub fn get_bus_id(&self) -> Option<u32> {
        match self {
            GPUSelector::BusId(bus_id) => Some(*bus_id),
            GPUSelector::Index(index) => get_device_bus_id_by_index(*index),
        }
    }

    pub fn get_device(&self) -> Option<&'static Device> {
        match self {
            GPUSelector::BusId(bus_id) => Device::by_bus_id(*bus_id).ok(),
            GPUSelector::Index(index) => get_device_by_index(*index),
        }
    }

    pub fn get_key(&self) -> String {
        match self {
            GPUSelector::BusId(id) => format!("BusID: {}", id),
            GPUSelector::Index(idx) => {
                if let Some(id) = self.get_bus_id() {
                    format!("BusID: {}", id)
                } else {
                    format!("Index: {}", idx)
                }
            }
        }
    }
}

fn get_device_bus_id_by_index(index: usize) -> Option<BusId> {
    if let Some(device) = get_device_by_index(index) {
        device.bus_id
    } else {
        None
    }
}

fn get_device_by_index(index: usize) -> Option<&'static Device> {
    Device::all_iter().nth(index)
}

pub struct Program {
    device_name: String,
    queue: CommandQueue,
    context: Context,
    kernels_by_name: HashMap<String, opencl3::kernel::Kernel>,
}

impl Program {
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

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

    pub fn create_kernel(&self, name: &str, gws: usize, lws: usize) -> GPUResult<Kernel> {
        let kernel = self
            .kernels_by_name
            .get(name)
            .ok_or_else(|| GPUError::KernelNotFound(name.to_string()))?;
        let mut builder = ExecuteKernel::new(&kernel);
        builder.set_global_work_size(gws);
        builder.set_local_work_size(lws);
        Ok(Kernel {
            builder,
            queue: &self.queue,
        })
    }

    pub fn write_from_buffer<T>(
        &self,
        buffer: &Buffer<T>,
        offset: usize,
        data: &[T],
    ) -> GPUResult<()> {
        assert!(offset + data.len() <= buffer.length);

        let buff = buffer
            .buffer
            .create_sub_buffer(CL_MEM_READ_WRITE, offset, data.len())?;

        self.queue
            .enqueue_write_buffer(&buff, CL_BLOCKING, 0, data, &[])?;

        Ok(())
    }

    pub fn read_into_buffer<T>(
        &self,
        buffer: &Buffer<T>,
        offset: usize,
        data: &mut [T],
    ) -> GPUResult<()> {
        assert!(offset + data.len() <= buffer.length);
        let buff = buffer
            .buffer
            .create_sub_buffer(CL_MEM_READ_WRITE, offset, data.len())?;

        self.queue
            .enqueue_read_buffer(&buff, CL_BLOCKING, 0, data, &[])?;

        Ok(())
    }

    /// Run some code in the context of the program
    ///
    /// On CUDA it sets the correct contexts and synchronizes the stream before returning.
    /// On OpenCL it's only executing the closure without any other side-effects.
    pub fn run<F, R, E>(&self, fun: F) -> Result<R, E>
    where
        F: FnOnce() -> Result<R, E>,
        E: From<GPUError>,
    {
        fun()
    }
}

pub trait KernelArgument<'a> {
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

pub struct LocalBuffer<T> {
    length: usize,
    _phantom: std::marker::PhantomData<T>,
}
impl<T> LocalBuffer<T> {
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

#[derive(Debug)]
pub struct Kernel<'a> {
    builder: ExecuteKernel<'a>,
    queue: &'a CommandQueue,
}

impl<'a> Kernel<'a> {
    pub fn arg<T: KernelArgument<'a>>(mut self, t: &T) -> Self {
        t.push(&mut self);
        self
    }
    pub fn run(mut self) -> GPUResult<()> {
        self.builder.enqueue_nd_range(&self.queue)?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::Device;

    #[test]
    fn test_device_all() {
        for _ in 0..10 {
            let devices = Device::all();
            dbg!(&devices.len());
        }
    }
}
