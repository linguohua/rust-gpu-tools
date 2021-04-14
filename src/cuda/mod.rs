mod error;
mod utils;

pub use error::*;
use sha2::{Digest, Sha256};
use std::convert::TryFrom;
use std::ffi::c_void;
use std::ffi::{CStr, CString};
use std::fmt::Write;
use std::hash::{Hash, Hasher};

use rustacuda::memory::AsyncCopyDestination;

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
    pub fn platform_name(&self) -> &'static str {
        match self {
            Brand::Nvidia => "NVIDIA CUDA",
            Brand::Amd => "AMD Accelerated Parallel Processing",
            Brand::Apple => "Apple",
        }
    }

    fn all() -> Vec<Brand> {
        vec![Brand::Nvidia, Brand::Amd, Brand::Apple]
    }
}

pub struct Buffer<T> {
    // TODO vmx 2021-04-14: make it private
    pub buffer: rustacuda::memory::DeviceBuffer<u8>,
    length: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Buffer<T> {
    /// The number of bytes / size_of(T)
    pub fn length(&self) -> usize {
        self.length
    }
}

#[derive(Debug, Clone)]
pub struct Device {
    brand: Brand,
    name: String,
    memory: u64,
    bus_id: Option<BusId>,
    pub device: rustacuda::device::Device,
    pub context: rustacuda::context::UnownedContext,
}

//unsafe impl Send for Device {}
// TODO vmx 2021-04-14: This is probably not a good idea an might actually be wrong. Do it for now
// to make the current code compile, but think about alternatives.
unsafe impl Sync for Device {}

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
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation
        Ok(true)
    }
    pub fn bus_id(&self) -> Option<BusId> {
        self.bus_id
    }

    /// Return all available GPU devices of supported brands, ordered by brand as
    /// defined by `Brand::all()`.
    pub fn all() -> Vec<&'static Device> {
        Self::all_iter().collect()
    }

    pub fn all_iter() -> impl Iterator<Item = &'static Device> {
        Brand::all()
            .into_iter()
            .filter_map(|brand| utils::DEVICES.get(&brand).map(|(devices, _)| devices))
            .flatten()
    }

    pub fn by_bus_id(bus_id: BusId) -> GPUResult<&'static Device> {
        Device::all_iter()
            .find(|d| match d.bus_id {
                Some(id) => bus_id == id,
                None => false,
            })
            .ok_or(GPUError::DeviceNotFound)
    }

    pub fn by_brand(brand: Brand) -> Option<&'static Vec<Device>> {
        utils::DEVICES.get(&brand).map(|(devices, _)| devices)
    }

    //pub fn cl_device_id(&self) -> cl_device_id {
    //    self.device.id()
    //}
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
            GPUSelector::BusId(bus_id) => Device::all_iter().find(|d| d.bus_id == Some(*bus_id)),
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

// TODO vmx 2021-02-26: Move to utils and re-export as here as public export
pub fn get_memory(d: &rustacuda::device::Device) -> GPUResult<u64> {
    match d.total_memory() {
        Ok(memory) => Ok(u64::try_from(memory).expect("Platform must be <= 64-bit")),
        Err(_) => Err(GPUError::DeviceInfoNotAvailable),
    }
}

pub struct Program {
    //device: rustacuda::device::Device,
    //context: rustacuda::context::UnownedContext,
    //device: Device,
    module: rustacuda::module::Module,
    stream: rustacuda::stream::Stream,
    //// TODO vmx 2021-04-11: The `Context` contains the devices, use those instead of storing them again
    //device: Device,
    //program: opencl3::program::Program,
    //queue: opencl3::command_queue::CommandQueue,
    //context: opencl3::context::Context,
    //kernels: HashMap<String, opencl3::kernel::Kernel>,
}

impl Program {
    //pub fn device(&self) -> Device {
    //    self.device.clone()
    //}
    // TODO vmx 2021-04-14: Think about if it makes it a nicer API if the filename won't be a CStr
    pub fn from_cuda(device: Device, filename: &CStr) -> GPUResult<Program> {
        //let cached = utils::cache_path(&device, src)?;
        //if std::path::Path::exists(&cached) {
        //    let bin = std::fs::read(cached)?;
        //    Program::from_binary(device, bin)
        //} else {
        //
        rustacuda::context::CurrentContext::set_current(&device.context)?;
        let module = rustacuda::module::Module::load_from_file(filename)?;
        let stream =
            rustacuda::stream::Stream::new(rustacuda::stream::StreamFlags::NON_BLOCKING, None)?;
        let prog = Program {
            // TODO vmx 2021-04-12: Decide whether to store a gpu tools `Device` or the CUDA device
            // and context separately
            //context: device.context.get_unowned(),
            //device: device.device,
            module,
            stream,
        };
        Ok(prog)
        //}
    }
    //pub fn from_binary(device: Device, bin: Vec<u8>) -> GPUResult<Program> {
    //    let context = opencl3::context::Context::from_device(&device.device)?;
    //    let bins = vec![&bin[..]];
    //    let program =
    //        opencl3::program::Program::create_from_binary(&context, context.devices(), &bins)?;
    //    program.build(context.devices(), "")?;
    //    let queue =
    //        opencl3::command_queue::CommandQueue::create(&context, context.default_device(), 0)?;
    //    let kernels = opencl3::kernel::create_program_kernels(&program)?;
    //    let kernels_by_name = kernels
    //        .into_iter()
    //        .map(|kernel| (kernel.function_name().unwrap(), kernel))
    //        .collect();
    //    Ok(Program {
    //        device,
    //        program,
    //        queue,
    //        context,
    //        kernels: kernels_by_name,
    //    })
    //}
    //pub fn to_binary(&self) -> GPUResult<Vec<u8>> {
    //    match self.program.get_binaries() {
    //        Ok(bins) => Ok(bins[0].clone()),
    //        Err(_) => Err(GPUError::ProgramInfoNotAvailable(
    //            opencl3::program::ProgramInfo::CL_PROGRAM_BINARIES,
    //        )),
    //    }
    //}
    pub fn create_buffer<T>(&self, length: usize) -> GPUResult<Buffer<T>> {
        assert!(length > 0);
        // TODO vmx 2021-04-12: Does the generic `T` and the `<u8>` make sense?
        let buffer = unsafe {
            rustacuda::memory::DeviceBuffer::<u8>::uninitialized(length * std::mem::size_of::<T>())?
        };

        Ok(Buffer::<T> {
            buffer,
            length,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn create_buffer_flexible<T>(&self, max_length: usize) -> GPUResult<Buffer<T>> {
        let mut curr = max_length;
        let mut step = max_length / 2;
        let mut n = 1;
        while step > 0 && n < max_length {
            if self.create_buffer::<T>(curr).is_ok() {
                n = curr;
                curr = std::cmp::min(curr + step, max_length);
            } else {
                curr -= step;
            }
            step /= 2;
        }
        self.create_buffer::<T>(n)
    }

    // TODO vmx 2021-04-14: perhaps rename to `prepare_kernel()` or `get_kernel()`, which seems
    // closer to what OpenCL and CUDA is doing.
    pub fn create_kernel(&self, name: &str, gws: usize, lws: Option<usize>) -> Kernel {
        let function_name = CString::new(name).expect("Kernel name must not contain nul byts");
        // TODO vmx 2021-04-14: That should be a proper error and not an `unwrap()`
        let function = self.module.get_function(&function_name).unwrap();

        Kernel {
            function,
            global_work_size: gws,
            local_work_size: lws,
            stream: &self.stream,
            args: Vec::new(),
        }
    }
    pub fn write_from_buffer<T>(
        &self,
        buffer: &mut Buffer<T>,
        offset: usize,
        data: &[T],
    ) -> GPUResult<()> {
        assert!(offset + data.len() <= buffer.length());
        unsafe {
            let bytes = std::slice::from_raw_parts(
                data.as_ptr() as *const T as *const u8,
                data.len() * std::mem::size_of::<T>(),
            );
            buffer.buffer.async_copy_from(bytes, &self.stream)?;
        };
        Ok(())
    }

    pub fn read_into_buffer<T>(
        &self,
        buffer: &Buffer<T>,
        offset: usize,
        data: &mut [T],
    ) -> GPUResult<()> {
        assert!(offset + data.len() <= buffer.length());

        unsafe {
            let bytes = std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut T as *mut u8,
                data.len() * std::mem::size_of::<T>(),
            );
            buffer.buffer.async_copy_to(bytes, &self.stream)?;
        };

        Ok(())
    }

    /// This is only needed for CUDA. Call this once you're done to synchronize the stream
    pub fn sync(&self) -> GPUResult<()> {
        self.stream.synchronize().map_err(Into::into)
    }
}

// TODO vmx 2021-04-14: Check if the kernel argument abstraction is still needed (it doesn't make
// much sense in CUDA and perhaps it's also no longer needed for OpenCL
pub trait KernelArgument<'a> {
    fn push(&self, kernel: &mut Kernel<'a>);
}

impl<'a, T> KernelArgument<'a> for &'a Buffer<T> {
    fn push(&self, kernel: &mut Kernel<'a>) {
        kernel
            .args
            .push(&self.buffer as *const _ as *mut ::std::ffi::c_void);
    }
}

impl KernelArgument<'_> for i32 {
    fn push(&self, kernel: &mut Kernel) {
        kernel
            .args
            .push(self as *const _ as *mut ::std::ffi::c_void);
    }
}

impl KernelArgument<'_> for u32 {
    fn push(&self, kernel: &mut Kernel) {
        kernel
            .args
            .push(self as *const _ as *mut ::std::ffi::c_void);
    }
}

//pub struct LocalBuffer<T> {
//    length: usize,
//    _phantom: std::marker::PhantomData<T>,
//}
//impl<T> LocalBuffer<T> {
//    pub fn new(length: usize) -> Self {
//        LocalBuffer::<T> {
//            length,
//            _phantom: std::marker::PhantomData,
//        }
//    }
//}
//
//impl<T> KernelArgument<'_> for LocalBuffer<T> {
//    fn push(&self, kernel: &mut Kernel) {
//        kernel
//            .builder
//            .set_arg_local_buffer::<T>(self.length * std::mem::size_of::<T>());
//    }
//}

//#[derive(Debug)]
pub struct Kernel<'a> {
    function: rustacuda::function::Function<'a>,
    //builder: opencl3::kernel::ExecuteKernel<'a>,
    //queue: &'a opencl3::command_queue::CommandQueue,
    global_work_size: usize,
    local_work_size: Option<usize>,
    stream: &'a rustacuda::stream::Stream,
    //args: Vec<rustacuda::memory::DevicePointer>,
    args: Vec<*mut c_void>,
}

impl<'a> Kernel<'a> {
    pub fn get_settings(
        &self,
    ) -> (
        &rustacuda::function::Function<'a>,
        usize,
        usize,
        &'a rustacuda::stream::Stream,
    ) {
        (
            &self.function,
            self.global_work_size,
            // TODO vmx 2021-04-14: Don't unwrap()
            self.local_work_size.unwrap(),
            self.stream,
        )
    }
    //pub fn arg<T: KernelArgument<'a>>(mut self, t: T) -> Self {
    //    t.push(&mut self);
    //    //self.args.push(t as *const _ as *mut ::std::ffi::c_void);
    //    self
    //}
    //pub fn run(mut self) -> GPUResult<()> {
    //            launch!(
    //            module.G2_bellman_multiexp<<<global_work_size as u32,
    //                                         LOCAL_WORK_SIZE as u32,
    //                                         0, stream>>>
    //                (base_buffer_g.as_device_ptr(),
    //                 bucket_buffer_g.as_device_ptr(),
    //                 result_buffer_g.as_device_ptr(),
    //                 exp_buffer_g.as_device_ptr(),
    //                 n as u32,
    //                 num_groups as u32,
    //                 num_windows as u32,
    //                 window_size as u32
    //                ))?;
    //    //let function = self.function
    //    launch!(
    //        self.function<<<self.global_work_size as u32, self.local_work_size.unwrap() as u32, 0 self.stream>>>(
    //
    //        )
    //    )?;
    //    Ok(())
    //}
}

#[macro_export]
macro_rules! call_cuda_kernel {
   ($kernel:expr, $( $arg:expr ),*) => {{
       let (function, global_work_size, local_work_size, stream) = $kernel.get_settings();
       unsafe {
           launch!(
               // TODO vmx 2021-04-14: Don't `unwrap()`, but make sure it's always set
               function<<<global_work_size as u32, local_work_size as u32, 0, stream>>>(
                   $($arg),*
               )
           )
       }
   }};
}
//#[macro_export]
//macro_rules! call_cuda_kernel {
//    ($kernel:expr, $($arg:expr),*) => {{
//        $kernel
//        $(.arg($arg))*
//        .run()
//    }};
//}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_device_all() {
        for _ in 0..10 {
            let devices = Device::all();
            dbg!(&devices.len());
        }
    }
}
