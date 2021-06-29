use std::convert::TryInto;

use log::{debug, warn};
use opencl3::device::DeviceInfo::CL_DEVICE_GLOBAL_MEM_SIZE;
use sha2::{Digest, Sha256};

use crate::device::{Brand, DeviceUuid, PciId};
use crate::error::{GPUError, GPUResult};
use crate::opencl::{Device, CL_UUID_SIZE_KHR};

const AMD_DEVICE_VENDOR_STRING: &str = "AMD";
const NVIDIA_DEVICE_VENDOR_STRING: &str = "NVIDIA Corporation";

fn get_pci_id(device: &opencl3::device::Device) -> GPUResult<PciId> {
    let vendor = device.vendor()?;
    let id = match vendor.as_ref() {
        AMD_DEVICE_VENDOR_STRING => {
            let topo = device.topology_amd()?;
            let device = topo.device as u32;
            let bus = topo.bus as u32;
            let function = topo.function as u32;
            (device << 16) | (bus << 8) | function
        }
        NVIDIA_DEVICE_VENDOR_STRING => device.pci_slot_id_nv()?,
        _ => return Err(GPUError::DevicePciId(vendor)),
    };
    Ok(id.into())
}

fn get_uuid(device: &opencl3::device::Device) -> GPUResult<DeviceUuid> {
    let uuid: [u8; CL_UUID_SIZE_KHR] = device
        .uuid_khr()?
        .try_into()
        .expect("opencl3 returned an invalid UUID");
    Ok(uuid.into())
}

pub fn cache_path(device: &Device, cl_source: &str) -> std::io::Result<std::path::PathBuf> {
    let path = dirs::home_dir().unwrap().join(".rust-gpu-tools");
    if !std::path::Path::exists(&path) {
        std::fs::create_dir(&path)?;
    }
    let mut hasher = Sha256::new();
    hasher.input(device.name.as_bytes());
    hasher.input(u32::from(device.pci_id).to_be_bytes());
    hasher.input(<[u8; CL_UUID_SIZE_KHR]>::from(
        device.uuid.unwrap_or_default(),
    ));
    hasher.input(cl_source.as_bytes());
    let filename = format!("{}.bin", hex::encode(hasher.result()));
    Ok(path.join(filename))
}

fn get_memory(d: &opencl3::device::Device) -> GPUResult<u64> {
    d.global_mem_size()
        .map_err(|_| GPUError::DeviceInfoNotAvailable(CL_DEVICE_GLOBAL_MEM_SIZE))
}

pub(crate) fn build_device_list() -> Vec<Device> {
    let mut all_devices = Vec::new();
    let platforms: Vec<_> = opencl3::platform::get_platforms().unwrap_or_default();

    for platform in platforms.iter() {
        let platform_name = match platform.name() {
            Ok(pn) => pn,
            Err(error) => {
                warn!("Cannot get platform name: {:?}", error);
                continue;
            }
        };
        if let Some(brand) = Brand::by_name(&platform_name) {
            let devices = platform
                .get_devices(opencl3::device::CL_DEVICE_TYPE_GPU)
                .map_err(Into::into)
                .and_then(|devices| {
                    devices
                        .into_iter()
                        .map(opencl3::device::Device::new)
                        .filter(|d| {
                            if let Ok(vendor) = d.vendor() {
                                match vendor.as_str() {
                                    // Only use devices from the accepted vendors ...
                                    AMD_DEVICE_VENDOR_STRING | NVIDIA_DEVICE_VENDOR_STRING => {
                                        // ... which are available.
                                        return d.available().unwrap_or(0) != 0;
                                    }
                                    _ => (),
                                }
                            }
                            false
                        })
                        .map(|d| -> GPUResult<_> {
                            Ok(Device {
                                brand,
                                name: d.name()?,
                                memory: get_memory(&d)?,
                                pci_id: get_pci_id(&d)?,
                                uuid: get_uuid(&d).ok(),
                                device: d,
                            })
                        })
                        .collect::<GPUResult<Vec<_>>>()
                });
            match devices {
                Ok(mut devices) => {
                    all_devices.append(&mut devices);
                }
                Err(err) => {
                    warn!("Unable to retrieve devices for {:?}: {:?}", brand, err);
                }
            }
        }
    }

    debug!("loaded devices: {:?}", all_devices);
    all_devices
}
