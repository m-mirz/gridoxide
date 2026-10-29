//! The CUDA runtime as gridoxide-gpu uses it: the generated bindings (cuDSS,
//! the CUDA runtime, and the kernels' launchers in `cuda/kernels.h`), error
//! conversion, and device buffers.

use std::ffi::{CStr, c_void};
use std::ptr;

use crate::uniform::LinearError;

#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    unused_imports,
    unnecessary_transmutes,
    clippy::all
)]
pub(crate) mod ffi {
    include!(concat!(env!("OUT_DIR"), "/cudss_bindings.rs"));
}

pub(crate) fn cuda(err: ffi::cudaError_t, what: &str) -> Result<(), LinearError> {
    if err == ffi::cudaSuccess {
        return Ok(());
    }
    // SAFETY: cudaGetErrorString returns a static, NUL-terminated string for
    // any error code, known or not.
    let msg = unsafe { CStr::from_ptr(ffi::cudaGetErrorString(err)) };
    Err(LinearError::Backend(format!(
        "{what}: {}",
        msg.to_string_lossy()
    )))
}

/// A device allocation of `len` values of `T`, freed on drop.
pub(crate) struct DeviceBuffer<T> {
    pub(crate) ptr: *mut T,
    len: usize,
}

impl<T: Copy> DeviceBuffer<T> {
    pub(crate) fn new(len: usize) -> Result<Self, LinearError> {
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: cudaMalloc writes a device pointer to `ptr` or fails.
        cuda(
            unsafe { ffi::cudaMalloc(&mut ptr, len * size_of::<T>()) },
            "cudaMalloc",
        )?;
        Ok(Self {
            ptr: ptr.cast(),
            len,
        })
    }

    /// Copies `src` to the device, ordered on `stream`. From pageable memory
    /// the call returns once `src` has been staged, so `src` may be reused.
    pub(crate) fn upload(
        &mut self,
        src: &[T],
        stream: ffi::cudaStream_t,
    ) -> Result<(), LinearError> {
        assert_eq!(src.len(), self.len);
        // SAFETY: both ranges are `len` values long.
        cuda(
            unsafe {
                ffi::cudaMemcpyAsync(
                    self.ptr.cast(),
                    src.as_ptr().cast(),
                    self.len * size_of::<T>(),
                    ffi::cudaMemcpyHostToDevice,
                    stream,
                )
            },
            "cudaMemcpyAsync (to device)",
        )
    }

    /// Copies the buffer into `dst` once everything queued on `stream`
    /// before it has run.
    pub(crate) fn download(
        &self,
        dst: &mut [T],
        stream: ffi::cudaStream_t,
    ) -> Result<(), LinearError> {
        assert_eq!(dst.len(), self.len);
        // SAFETY: both ranges are `len` values long; the synchronize makes
        // the copy complete before `dst` is read.
        cuda(
            unsafe {
                ffi::cudaMemcpyAsync(
                    dst.as_mut_ptr().cast(),
                    self.ptr.cast(),
                    self.len * size_of::<T>(),
                    ffi::cudaMemcpyDeviceToHost,
                    stream,
                )
            },
            "cudaMemcpyAsync (to host)",
        )?;
        cuda(
            unsafe { ffi::cudaStreamSynchronize(stream) },
            "cudaStreamSynchronize",
        )
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `ptr` came from cudaMalloc and is freed once.
            unsafe { ffi::cudaFree(self.ptr.cast()) };
        }
    }
}

impl<T> DeviceBuffer<T> {
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

/// Copies `src` to `dst` on the device, ordered on `stream`, as
/// [`DeviceBuffer::upload`] does into memory it does not own.
///
/// # Safety
///
/// `dst` must be a device allocation of at least `src.len()` values.
pub(crate) unsafe fn copy_to_device<T: Copy>(
    dst: *mut T,
    src: &[T],
    stream: ffi::cudaStream_t,
) -> Result<(), LinearError> {
    // SAFETY: the caller's.
    cuda(
        unsafe {
            ffi::cudaMemcpyAsync(
                dst.cast(),
                src.as_ptr().cast(),
                std::mem::size_of_val(src),
                ffi::cudaMemcpyHostToDevice,
                stream,
            )
        },
        "cudaMemcpyAsync (to device)",
    )
}

/// A CUDA stream, synchronized and destroyed on drop. Declared before the
/// buffers in a struct that owns both, it is dropped first, so no queued
/// work outlives the memory it uses.
pub(crate) struct Stream(pub(crate) ffi::cudaStream_t);

impl Stream {
    pub(crate) fn new() -> Result<Self, LinearError> {
        let mut stream: ffi::cudaStream_t = ptr::null_mut();
        // SAFETY: cudaStreamCreate writes a stream handle or fails.
        cuda(
            unsafe { ffi::cudaStreamCreate(&mut stream) },
            "cudaStreamCreate",
        )?;
        Ok(Self(stream))
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: the stream was created in `new` and is destroyed once.
        unsafe {
            ffi::cudaStreamSynchronize(self.0);
            ffi::cudaStreamDestroy(self.0);
        }
    }
}
