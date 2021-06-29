use std::convert::TryFrom;

use log::{debug, warn};

use crate::cuda::Device;
use crate::device::{Brand, DeviceUuid, PciId};
use crate::error::{GPUError, GPUResult};

#[repr(C)]
#[derive(Debug, Clone, Default)]
struct cl_amd_device_topology {
    r#type: u32,
    unused: [u8; 17],
    bus: u8,
    device: u8,
    function: u8,
}

fn get_pci_id(device: &rustacuda::device::Device) -> Result<PciId, GPUError> {
    match device.get_attribute(rustacuda::device::DeviceAttribute::PciDeviceId) {
        Ok(pci_id) => Ok(PciId::from(pci_id as u32)),
        Err(error) => Err(error.into()),
    }
}

fn get_uuid(_device: &rustacuda::device::Device) -> Result<DeviceUuid, GPUError> {
    // TODO vmx 2021-06-28: Needs an implementation in RUSTACUDA, use
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g987b46b884c101ed5be414ab4d9e60e4
    // for that, which is already part of cuda-sys.
    Err(GPUError::DeviceNotFound)
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

fn get_memory(d: &rustacuda::device::Device) -> GPUResult<u64> {
    let memory = d.total_memory()?;
    Ok(u64::try_from(memory).expect("Platform must be <= 64-bit"))
}

pub(crate) fn build_device_list() -> (Vec<Device>, CudaContexts) {
    rustacuda::init(rustacuda::CudaFlags::empty())
        .map(|_| {
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
                                rustacuda::context::ContextStack::pop().unwrap();

                                (
                                    Device {
                                        brand: Brand::Nvidia,
                                        // TODO vmx 2021-04-14: Look why this could fail
                                        name: device.name().unwrap(),
                                        // TODO vmx 2021-04-14: Look why this could fail
                                        memory: get_memory(&device).unwrap(),
                                        pci_id: get_pci_id(&device).unwrap(),
                                        uuid: get_uuid(&device).ok(),
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

            let (devices, contexts) = devices_and_contexts.into_iter().unzip();
            let wrapped_contexts = CudaContexts(contexts);

            debug!("loaded devices: {:?}", devices);
            (devices, wrapped_contexts)
        })
        .unwrap_or_else(|err| {
            warn!("failed to init cuda: {:?}", err);
            (Vec::new(), CudaContexts(Vec::new()))
        })
}
