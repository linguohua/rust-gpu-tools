use std::fmt;

use lazy_static::lazy_static;
use log::debug;
#[cfg(all(feature = "opencl", feature = "cuda"))]
use log::warn;

use std::convert::TryFrom;
use std::hash::{Hash, Hasher};
use std::mem;

use crate::error::{GPUError, GPUResult};

#[cfg(feature = "cuda")]
use crate::cuda;
#[cfg(feature = "opencl")]
use crate::opencl;

pub const CL_UUID_SIZE_KHR: usize = 16;

#[cfg(feature = "cuda")]
lazy_static! {
    // The owned CUDA contexts are stored globally. Each devives contains an unowned reference,
    // so that devices can be cloned.
    static ref DEVICES: (Vec<Device>, cuda::utils::CudaContexts) = build_device_list();
}

#[cfg(all(feature = "opencl", not(feature = "cuda")))]
lazy_static! {
    // Keep it as a tuple as the CUDA case, so that the using `DEVICES` is independent of the
    // features set.
    static ref DEVICES: (Vec<Device>, ()) = build_device_list();
}

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

// Only list supported brands
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Brand {
    Amd,
    Nvidia,
}

impl Brand {
    /// Returns a brand by name if it exists
    pub fn by_name(name: &str) -> Option<Self> {
        match name {
            "NVIDIA CUDA" => Some(Self::Nvidia),
            "AMD Accelerated Parallel Processing" => Some(Self::Amd),
            _ => None,
        }
    }
}

impl fmt::Display for Brand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let brand = match self {
            Brand::Nvidia => "NVIDIA CUDA",
            Brand::Amd => "AMD Accelerated Parallel Processing",
        };
        write!(f, "{}", brand)
    }
}

#[derive(Debug)]
pub enum Framework {
    Cuda,
    Opencl,
}

#[derive(Debug, Clone)]
pub struct Device {
    brand: Brand,
    name: String,
    memory: u64,
    // All devices have a PCI ID. It is used as fallback in case there is not UUID.
    pci_id: PciId,
    uuid: Option<DeviceUuid>,
    #[cfg(feature = "opencl")]
    opencl: Option<opencl::Device>,
    #[cfg(feature = "cuda")]
    cuda: Option<cuda::Device>,
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

    /// Returns the best possible unique identifier, a UUID is preferred over a PCI ID.
    pub fn unique_id(&self) -> UniqueId {
        match self.uuid {
            Some(uuid) => UniqueId::Uuid(uuid),
            None => UniqueId::PciId(self.pci_id),
        }
    }

    /// Returns the preferred framework (CUDA or OpenCL) to use.
    ///
    /// CUDA will be be preferred over OpenCL. The returned framework will work on the device.
    /// E.g. it won't return `Framework::Cuda` for an AMD device.
    pub fn framework(&self) -> Framework {
        #[cfg(all(feature = "opencl", feature = "cuda"))]
        if cfg!(feature = "cuda") && self.cuda.is_some() {
            Framework::Cuda
        } else {
            Framework::Opencl
        }

        #[cfg(all(feature = "cuda", not(feature = "opencl")))]
        {
            Framework::Cuda
        }

        #[cfg(all(feature = "opencl", not(feature = "cuda")))]
        {
            Framework::Opencl
        }
    }

    /// Returns the underlying OpenCL device if it is available.
    #[cfg(feature = "opencl")]
    pub fn opencl_device(&self) -> Option<&opencl::Device> {
        self.opencl.as_ref()
    }

    /// Returns the underlying CUDA device if it is available.
    #[cfg(feature = "cuda")]
    pub fn cuda_device(&self) -> Option<&cuda::Device> {
        self.cuda.as_ref()
    }

    /// Returns all available GPU devices of supported brands.
    pub fn all() -> Vec<&'static Device> {
        Self::all_iter().collect()
    }

    /// Returns the device matching the PCI ID if there is one.
    pub fn by_pci_id(pci_id: PciId) -> Option<&'static Device> {
        Self::all_iter().find(|d| pci_id == d.pci_id)
    }

    /// Returns the device matching the UUID if there is one.
    pub fn by_uuid(uuid: DeviceUuid) -> Option<&'static Device> {
        Self::all_iter().find(|d| Some(uuid) == d.uuid)
    }

    /// Returns the device matching the unique ID if there is one.
    pub fn by_unique_id(unique_id: UniqueId) -> Option<&'static Device> {
        Self::all_iter().find(|d| unique_id == d.unique_id())
    }

    fn all_iter() -> impl Iterator<Item = &'static Device> {
        DEVICES.0.iter()
    }
}

#[cfg(feature = "cuda")]
fn build_device_list() -> (Vec<Device>, cuda::utils::CudaContexts) {
    let mut all_devices = Vec::new();

    #[cfg(feature = "opencl")]
    let opencl_devices = opencl::utils::build_device_list();

    #[cfg(all(feature = "cuda", feature = "opencl"))]
    let (mut cuda_devices, cuda_contexts) = cuda::utils::build_device_list();
    #[cfg(all(feature = "cuda", not(feature = "opencl")))]
    let (cuda_devices, cuda_contexts) = cuda::utils::build_device_list();

    // Combine OpenCL and CUDA devices into one device if it is the same GPU
    #[cfg(feature = "opencl")]
    for opencl_device in opencl_devices {
        let mut device = Device {
            brand: opencl_device.brand(),
            name: opencl_device.name(),
            memory: opencl_device.memory(),
            pci_id: opencl_device.pci_id(),
            uuid: opencl_device.uuid(),
            opencl: Some(opencl_device),
            cuda: None,
        };

        // Only devices from Nvidia can use CUDA
        #[cfg(feature = "cuda")]
        if device.brand == Brand::Nvidia {
            for ii in 0..cuda_devices.len() {
                if (device.uuid.is_some() && cuda_devices[ii].uuid() == device.uuid)
                    || (cuda_devices[ii].pci_id() == device.pci_id)
                {
                    if device.memory() != cuda_devices[ii].memory() {
                        warn!("OpenCL and CUDA report different amounts of memory for a device with the same identifier");
                        continue;
                    }
                    // Move the CUDA device out of the vector
                    device.cuda = Some(cuda_devices.remove(ii));
                    // Only one device can match
                    continue;
                }
            }
        }

        all_devices.push(device)
    }

    // All CUDA devices that don't have a corresponding OpenCL devices
    for cuda_device in cuda_devices {
        let device = Device {
            brand: cuda_device.brand(),
            name: cuda_device.name(),
            memory: cuda_device.memory(),
            pci_id: cuda_device.pci_id(),
            uuid: cuda_device.uuid(),
            #[cfg(feature = "opencl")]
            opencl: None,
            cuda: Some(cuda_device),
        };
        all_devices.push(device);
    }

    debug!("loaded devices: {:?}", all_devices);
    (all_devices, cuda_contexts)
}

#[cfg(all(feature = "opencl", not(feature = "cuda")))]
fn build_device_list() -> (Vec<Device>, ()) {
    let devices = opencl::utils::build_device_list()
        .into_iter()
        .map(|device| Device {
            brand: device.brand(),
            name: device.name(),
            memory: device.memory(),
            pci_id: device.pci_id(),
            uuid: device.uuid(),
            opencl: Some(device),
        })
        .collect();

    debug!("loaded devices: {:?}", devices);
    (devices, ())
}

#[cfg(test)]
mod test {
    use super::{Device, DeviceUuid, PciId, UniqueId};
    use std::convert::TryFrom;

    #[test]
    fn test_device_all() {
        let devices = Device::all();
        for device in devices.iter() {
            println!("device: {:?}", device);
        }
        assert!(!devices.is_empty());
    }

    #[test]
    fn test_uuid() {
        let valid_string = "46abccd6-022e-b783-572d-833f7104d05f";
        let valid = DeviceUuid::try_from(valid_string).unwrap();
        assert_eq!(valid_string, &valid.to_string());

        let too_short_string = "ccd6-022e-b783-572d-833f7104d05f";
        let too_short = DeviceUuid::try_from(too_short_string);
        assert!(too_short.is_err(), "Parse error when UUID is too short.");

        let invalid_hex_string = "46abccd6-022e-b783-572d-833f7104d05h";
        let invalid_hex = DeviceUuid::try_from(invalid_hex_string);
        assert!(
            invalid_hex.is_err(),
            "Parse error when UUID containts non-hex character."
        );
    }

    #[test]
    fn test_pci_id() {
        let valid_string = "3f7104d0";
        let valid = PciId::try_from(valid_string).unwrap();
        assert_eq!(valid_string, &valid.to_string());

        let too_short_string = "3f71";
        let too_short = PciId::try_from(too_short_string);
        assert!(too_short.is_err(), "Parse error when PCI ID is too short.");

        let invalid_hex_string = "aabbxxdd";
        let invalid_hex = PciId::try_from(invalid_hex_string);
        assert!(
            invalid_hex.is_err(),
            "Parse error when PCI ID containts non-hex character."
        );
    }

    #[test]
    fn test_unique_id() {
        let valid_pci_id_string = "aabbccdd";
        let valid_pci_id = UniqueId::try_from(valid_pci_id_string).unwrap();
        assert_eq!(valid_pci_id_string, &valid_pci_id.to_string());

        let valid_uuid_string = "aabbccdd-eeff-0011-2233-445566778899";
        let valid_uuid = UniqueId::try_from(valid_uuid_string).unwrap();
        assert_eq!(valid_uuid_string, &valid_uuid.to_string());

        let invalid_string = "aabbccddeeffgg";
        let invalid = UniqueId::try_from(invalid_string);
        assert!(
            invalid.is_err(),
            "Parse error when ID matches neither a PCI Id, nor a UUID."
        );
    }
}
