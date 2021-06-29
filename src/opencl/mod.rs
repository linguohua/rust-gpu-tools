mod error;
mod utils;

use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;
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

pub const CL_UUID_SIZE_KHR: usize = 16;

#[allow(non_camel_case_types)]
pub type cl_device_id = opencl3::types::cl_device_id;

/// A unique identifier based on the PCI information.
#[derive(Debug, Copy, Clone, PartialEq, Hash)]
pub struct PciId(u32);

impl From<u32> for PciId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<PciId> for u32 {
    fn from(id: PciId) -> Self {
        id.0
    }
}

impl TryFrom<&str> for PciId {
    type Error = GPUError;

    fn try_from(pci_id: &str) -> GPUResult<Self> {
        let mut bytes = [0; mem::size_of::<u32>()];
        hex::decode_to_slice(pci_id, &mut bytes).map_err(|_| {
            GPUError::ParseId(format!(
                "Cannot parse PCI ID, expected hex-encoded string formated as aabbccdd, got {0}.",
                pci_id
            ))
        })?;
        let parsed = u32::from_le_bytes(bytes);
        Ok(Self(parsed))
    }
}

impl fmt::Display for PciId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = u32::to_le_bytes(self.0);
        // Formats the PCI ID as hex value, e.g: 46abccd6
        write!(f, "{}", hex::encode(&bytes[..4]),)
    }
}

/// A unique identifier based on UUID of the device.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
pub struct DeviceUuid([u8; CL_UUID_SIZE_KHR]);

impl From<[u8; CL_UUID_SIZE_KHR]> for DeviceUuid {
    fn from(uuid: [u8; CL_UUID_SIZE_KHR]) -> Self {
        Self(uuid)
    }
}

impl From<DeviceUuid> for [u8; CL_UUID_SIZE_KHR] {
    fn from(uuid: DeviceUuid) -> Self {
        uuid.0
    }
}

impl TryFrom<&str> for DeviceUuid {
    type Error = GPUError;

    fn try_from(uuid: &str) -> GPUResult<Self> {
        let mut bytes = [0; CL_UUID_SIZE_KHR];
        hex::decode_to_slice(uuid.replace("-", ""), &mut bytes)
            .map_err(|_| {
                GPUError::ParseId(format!("Cannot parse UUID, expected hex-encoded string formated as aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, got {0}.", uuid))
            })?;
        Ok(Self(bytes))
    }
}

impl fmt::Display for DeviceUuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Formats the uuid the same way as clinfo does, as an example:
        // the output should looks like 46abccd6-022e-b783-572d-833f7104d05f
        write!(
            f,
            "{}-{}-{}-{}-{}",
            hex::encode(&self.0[..4]),
            hex::encode(&self.0[4..6]),
            hex::encode(&self.0[6..8]),
            hex::encode(&self.0[8..10]),
            hex::encode(&self.0[10..])
        )
    }
}

impl fmt::Debug for DeviceUuid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

/// Unique identifier that can either be a PCI ID or a UUID.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum UniqueId {
    PciId(PciId),
    Uuid(DeviceUuid),
}

/// If the string contains a dash, it's interpreted as UUID, else it's interpreted as PCI ID.
impl TryFrom<&str> for UniqueId {
    type Error = GPUError;

    fn try_from(unique_id: &str) -> GPUResult<Self> {
        Ok(match unique_id.contains('-') {
            true => Self::Uuid(DeviceUuid::try_from(unique_id)?),
            false => Self::PciId(PciId::try_from(unique_id)?),
        })
    }
}

impl fmt::Display for UniqueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PciId(id) => id.fmt(f),
            Self::Uuid(id) => id.fmt(f),
        }
    }
}

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
    pub fn pci_id(&self) -> PciId {
        self.pci_id
    }

    pub fn uuid(&self) -> Option<DeviceUuid> {
        self.uuid
    }

    /// Returns the best possible unique identifier, a UUID is preferred over a PCI ID.
    pub fn unique_id(&self) -> UniqueId {
        match self.uuid {
            Some(uuid) => UniqueId::Uuid(uuid),
            None => UniqueId::PciId(self.pci_id),
        }
    }

    /// Return all available GPU devices of supported brands.
    pub fn all() -> Vec<&'static Device> {
        Self::all_iter().collect()
    }

    pub fn by_pci_id(pci_id: PciId) -> GPUResult<&'static Device> {
        Self::all_iter()
            .find(|d| pci_id == d.pci_id)
            .ok_or(GPUError::DeviceNotFound)
    }

    pub fn by_uuid(uuid: DeviceUuid) -> GPUResult<&'static Device> {
        Self::all_iter()
            .find(|d| match d.uuid {
                Some(id) => uuid == id,
                None => false,
            })
            .ok_or(GPUError::DeviceNotFound)
    }

    pub fn by_unique_id(unique_id: UniqueId) -> GPUResult<&'static Device> {
        Self::all_iter()
            .find(|d| unique_id == d.unique_id())
            .ok_or(GPUError::DeviceNotFound)
    }

    fn all_iter() -> impl Iterator<Item = &'static Device> {
        utils::DEVICES.iter()
    }

    pub fn cl_device_id(&self) -> cl_device_id {
        self.device.id()
    }
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

    /// Returns a kernel.
    ///
    /// The `global_work_size` does *not* follow the OpenCL definition. It is *not* the total
    /// number of threads. Instead it follows CUDA's definition and is the number of
    /// `local_work_size` sized thread groups.
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
