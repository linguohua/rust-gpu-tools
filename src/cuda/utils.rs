use std::collections::HashMap;

use cl3::{api_info_value, api_info_vector};
use lazy_static::lazy_static;
use log::{debug, warn};
use opencl3::error_codes::{ClError, CL_SUCCESS};
use opencl3::types::{cl_int, cl_uint};
// TODO vmx 2021-02-26: Don't use cl_sys directly, but implement it in opecl3/cl3
use cl_sys::clGetDeviceInfo;
// TODO vmx 2021-02-26: This is needed for the `api_info_value` macro. Change the macro itself
// to use `std::mem` instead of `mem`. This won't be needed in case that `api_info_value`
// invocation gets directly implemented in `cl3`
use std::mem;

use super::*;

#[repr(C)]
#[derive(Debug, Clone, Default)]
struct cl_amd_device_topology {
    r#type: u32,
    unused: [u8; 17],
    bus: u8,
    device: u8,
    function: u8,
}

pub fn get_bus_id(d: &rustacuda::device::Device) -> Result<u32, GPUError> {
    match d.get_attribute(rustacuda::device::DeviceAttribute::PciBusId) {
        Ok(pci_bus_id) => Ok(pci_bus_id as u32),
        Err(error) => Err(error.into()),
    }
}

pub fn cache_path(device: &Device, cl_source: &str) -> std::io::Result<std::path::PathBuf> {
    let path = dirs::home_dir().unwrap().join(".rust-gpu-tools");
    if !std::path::Path::exists(&path) {
        std::fs::create_dir(&path)?;
    }
    let mut hasher = Sha256::new();
    // If there are multiple devices with the same name and neither has a Bus-Id,
    // then there will be a collision. Bus-Id can be missing in the case of an Apple
    // GPU. For now, we assume that in the unlikely event of a collision, the same
    // cache can be used.
    // TODO: We might be able to get around this issue by using cl_vendor_id instead of Bus-Id.
    hasher.input(device.name.as_bytes());
    if let Some(bus_id) = device.bus_id {
        hasher.input(bus_id.to_be_bytes());
    }
    hasher.input(cl_source.as_bytes());
    let mut digest = String::new();
    for &byte in hasher.result()[..].iter() {
        write!(&mut digest, "{:x}", byte).unwrap();
    }
    write!(&mut digest, ".bin").unwrap();

    Ok(path.join(digest))
}

// NOTE vmx 2021-04-14: This is a dirty hack to make sure contexts stay around. We wrap them, so
// that `Sync` can be implemented. `Sync` is needed for lazy static. Though those contexts are
// never used directly, they are only accessed through the device which contains an
// `UnownedContext`. The device cannot contain the context itself, as then it couldn't be cloned,
// which is needed for creating the kernels.
// TODO vmx 2021-04-14: Check if perhaps the kernels could be created with a reference to the
// devices instead of owning them.
pub struct CudaContexts(Vec<rustacuda::context::Context>);
unsafe impl Sync for CudaContexts {}

lazy_static! {
    pub static ref PLATFORMS: Vec<opencl3::platform::Platform> =
        opencl3::platform::get_platforms().unwrap_or_default();
    //pub static ref DEVICES: HashMap<Brand, Vec<(Device, rustacuda::context::Context)>> = build_device_list();
    pub static ref DEVICES: HashMap<Brand, (Vec<Device>, CudaContexts)> = build_device_list();
}

pub fn find_platform(platform_name: &str) -> Result<Option<&opencl3::platform::Platform>, cl_int> {
    let platform = PLATFORMS.iter().find(|&p| match p.clone().name() {
        Ok(p) => p == platform_name,
        Err(_) => false,
    });
    Ok(platform)
}

//fn build_device_list() -> HashMap<Brand, Vec<(Device, rustacuda::context::Context)>> {
fn build_device_list() -> HashMap<Brand, (Vec<Device>, CudaContexts)> {
    let devices = rustacuda::device::Device::devices();
    println!("vmx: rust gpu tools: utils: build_device_list: {:?}", devices);
    // TODO vmx 20221-04-14: no `unwrpa()`, but proper error handling
    let devices_and_contexts: Vec<(Device, rustacuda::context::Context)> =
        rustacuda::device::Device::devices()
            .unwrap()
            .map(|device| {
                match device {
                    Ok(device) => {
                        // TODO vmx 2021-04-14: proper error handling
                        let context = rustacuda::context::Context::create_and_push(
                            rustacuda::context::ContextFlags::MAP_HOST
                                | rustacuda::context::ContextFlags::SCHED_AUTO,
                            device,
                        )
                        .unwrap();

                        (
                            Device {
                                brand: Brand::Nvidia,
                                // TODO vmx 2021-04-14: Look why this could fail
                                name: device.name().unwrap(),
                                // TODO vmx 2021-04-14: Look why this could fail
                                memory: get_memory(&device).unwrap(),
                                bus_id: get_bus_id(&device).ok(),
                                device,
                                context: context.get_unowned(),
                            },
                            context,
                        )
                    }
                    Err(_) => panic!("TODO some proper error"),
                }
            })
            .collect();

    //let (devices, contexts): (Vec<Device>, Vec<rustacuda::context::Context>) = devicesAndContexts.into_iter().unzip();
    let (devices, contexts) = devices_and_contexts.into_iter().unzip();
    let wrapped_contexts = CudaContexts(contexts);

    debug!("loaded devices: {:?}", devices);
    let mut device_list = HashMap::with_capacity(1);
    device_list.insert(Brand::Nvidia, (devices, wrapped_contexts));
    device_list

    //let brands = Brand::all();
    //let mut map = HashMap::with_capacity(brands.len());
    //
    //for brand in brands.into_iter() {
    //    match find_platform(brand.platform_name()) {
    //        Ok(Some(platform)) => {
    //            let devices = platform
    //                .get_devices(opencl3::device::CL_DEVICE_TYPE_ALL)
    //                .map_err(Into::into)
    //                .and_then(|devices| {
    //                    devices
    //                        .into_iter()
    //                        .map(opencl3::device::Device::new)
    //                        //.filter(|d| {
    //                        //    if let Ok(vendor) = d.vendor() {
    //                        //        match vendor.as_str() {
    //                        //            // Only use devices from the accepted vendors ...
    //                        //            AMD_DEVICE_VENDOR_STRING | NVIDIA_DEVICE_VENDOR_STRING => {
    //                        //                // ... which are available.
    //                        //                return d.available().unwrap_or(0) != 0;
    //                        //            }
    //                        //            _ => (),
    //                        //        }
    //                        //    }
    //                        //    false
    //                        //})
    //                        .map(|d| -> GPUResult<_> {
    //                            Ok(Device {
    //                                brand,
    //                                name: d.name()?,
    //                                memory: get_memory(&d)?,
    //                                bus_id: utils::get_bus_id(&d).ok(),
    //                                device: d,
    //                            })
    //                        })
    //                        .collect::<GPUResult<Vec<_>>>()
    //                });
    //            match devices {
    //                Ok(devices) => {
    //                    map.insert(brand, devices);
    //                }
    //                Err(err) => {
    //                    warn!("Unable to retrieve devices for {:?}: {:?}", brand, err);
    //                }
    //            }
    //        }
    //        Ok(None) => {}
    //        Err(err) => {
    //            warn!("Platform issue for brand {:?}: {:?}", brand, err);
    //        }
    //    }
    //}
}
