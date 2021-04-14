use rustacuda::error::CudaError;

#[derive(thiserror::Error, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum GPUError {
    #[error("Cuda Error: {0}")]
    Cuda(#[from] CudaError),
    #[error("Device not found!")]
    DeviceNotFound,
    #[error("Device info not available!")]
    DeviceInfoNotAvailable,
    //#[error("Device info not available!")]
    //DeviceInfoNotAvailable(DeviceInfo),
    //#[error("Program info not available!")]
    //ProgramInfoNotAvailable(ProgramInfo),
    #[error("IO Error: {0}")]
    IO(#[from] std::io::Error),
    #[error("Cannot get bus ID for device with vendor {0}")]
    DeviceBusId(String),
}

#[allow(clippy::upper_case_acronyms)]
#[allow(dead_code)]
pub type GPUResult<T> = std::result::Result<T, GPUError>;

//impl From<CudaError> for GPUError {
//    fn from(error: CudaError) -> Self {
//        GPUError::Cuda(error)
//    }
//}
