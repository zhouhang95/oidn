// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Locks the device that owns the specified object and saves/restores state
// Use *only* inside OIDN_TRY/CATCH!
#define OIDN_LOCK(obj) \
  DeviceGuard guard(obj);

// Try/catch for converting exceptions to errors
#define OIDN_TRY \
  try {

#define OIDN_CATCH(obj) \
  } catch (const Exception& e) {                                                                    \
    Device::setError(obj ? obj->getDevice() : nullptr, e.code(), e.what());                         \
  } catch (const std::bad_alloc&) {                                                                 \
    Device::setError(obj ? obj->getDevice() : nullptr, Error::OutOfMemory, "out of memory");        \
  } catch (const std::exception& e) {                                                               \
    Device::setError(obj ? obj->getDevice() : nullptr, Error::Unknown, e.what());                   \
  } catch (...) {                                                                                   \
    Device::setError(obj ? obj->getDevice() : nullptr, Error::Unknown, "unknown exception caught"); \
  }

#include "common/common.h"
#include "core/context.h"
#include "core/filter.h"
#include <mutex>

OIDN_NAMESPACE_USING
OIDN_API_NAMESPACE_BEGIN

  class DeviceGuard
  {
  public:
    template<typename T>
    DeviceGuard(T* obj)
      : device(obj->getDevice()),
        lock(device->getMutex())
    {
      device->begin(); // save state
    }

    ~DeviceGuard()
    {
      device->end(); // restore state
    }

  private:
    Device* device;
    std::lock_guard<std::mutex> lock;
  };

  namespace
  {
    OIDN_INLINE void checkHandle(void* handle)
    {
      if (handle == nullptr)
        throw Exception(Error::InvalidArgument, "invalid handle");
    }

    template<typename T>
    OIDN_INLINE void retainObject(T* obj)
    {
      if (obj)
      {
        obj->incRef();
      }
      else
      {
        OIDN_TRY
          checkHandle(obj);
        OIDN_CATCH(obj)
      }
    }

    template<typename T>
    OIDN_INLINE void releaseObject(T* obj)
    {
      if (obj == nullptr || obj->decRefKeep() == 0)
      {
        OIDN_TRY
          checkHandle(obj);
          OIDN_LOCK(obj);
          obj->getDevice()->wait(); // wait for all async operations to complete
          obj->destroy();
        OIDN_CATCH(obj)
      }
    }

    template<>
    OIDN_INLINE void releaseObject(Device* obj)
    {
      if (obj == nullptr || obj->decRefKeep() == 0)
      {
        OIDN_TRY
          checkHandle(obj);
          // Do NOT lock the device because it owns the mutex
          obj->begin(); // save stase
          obj->wait();  // wait for all async operations to complete
          obj->end();   // restore state
          obj->destroy();
        OIDN_CATCH(obj)
      }
    }
  }

  OIDN_API OIDNDevice oidnNewDevice(OIDNDeviceType inType)
  {
    DeviceType type = static_cast<DeviceType>(inType);
    Ref<Device> device = nullptr;

    OIDN_TRY
      Context& ctx = Context::get();

      if (type == DeviceType::Default)
      {
        for (auto curType : {DeviceType::CUDA, DeviceType::HIP, DeviceType::SYCL, DeviceType::CPU})
        {
          if (ctx.isDeviceSupported(curType))
          {
            type = curType;
            break;
          }
        }
      }

      device = ctx.getDeviceFactory(type)->newDevice();
    OIDN_CATCH(device)
    
    return reinterpret_cast<OIDNDevice>(device.detach());
  }

  OIDN_API OIDNDevice oidnNewSYCLDevice(const sycl::queue* queues, int numQueues)
  {
    Ref<Device> device = nullptr;
    OIDN_TRY
      Context& ctx = Context::get();
      auto factory = static_cast<SYCLDeviceFactoryBase*>(ctx.getDeviceFactory(DeviceType::SYCL));
      device = factory->newDevice(queues, numQueues);
    OIDN_CATCH(device)
    return reinterpret_cast<OIDNDevice>(device.detach());
  }

  OIDN_API OIDNDevice oidnNewCUDADevice(const int* deviceIds, const cudaStream_t* streams, int num)
  {
    Ref<Device> device = nullptr;
    OIDN_TRY
      Context& ctx = Context::get();
      auto factory = static_cast<CUDADeviceFactoryBase*>(ctx.getDeviceFactory(DeviceType::CUDA));
      device = factory->newDevice(deviceIds, streams, num);
    OIDN_CATCH(device)
    return reinterpret_cast<OIDNDevice>(device.detach());
  }

  OIDN_API OIDNDevice oidnNewHIPDevice(const int* deviceIds, const hipStream_t* streams, int num)
  {
    Ref<Device> device = nullptr;
    OIDN_TRY
      Context& ctx = Context::get();
      auto factory = static_cast<HIPDeviceFactoryBase*>(ctx.getDeviceFactory(DeviceType::HIP));
      device = factory->newDevice(deviceIds, streams, num);
    OIDN_CATCH(device)
    return reinterpret_cast<OIDNDevice>(device.detach());
  }

  OIDN_API void oidnRetainDevice(OIDNDevice hDevice)
  {
    Device* device = reinterpret_cast<Device*>(hDevice);
    retainObject(device);
  }

  OIDN_API void oidnReleaseDevice(OIDNDevice hDevice)
  {
    Device* device = reinterpret_cast<Device*>(hDevice);
    releaseObject(device);
  }

  OIDN_API void oidnSetDevice1b(OIDNDevice hDevice, const char* name, bool value)
  {
    Device* device = reinterpret_cast<Device*>(hDevice);
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      device->set1i(name, value);
    OIDN_CATCH(device)
  }

  OIDN_API void oidnSetDevice1i(OIDNDevice hDevice, const char* name, int value)
  {
    Device* device = reinterpret_cast<Device*>(hDevice);
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      device->set1i(name, value);
    OIDN_CATCH(device)
  }

  OIDN_API bool oidnGetDevice1b(OIDNDevice hDevice, const char* name)
  {
    Device* device = reinterpret_cast<Device*>(hDevice);
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      return device->get1i(name);
    OIDN_CATCH(device)
    return false;
  }

  OIDN_API int oidnGetDevice1i(OIDNDevice hDevice, const char* name)
  {
    Device* device = reinterpret_cast<Device*>(hDevice);
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      return device->get1i(name);
    OIDN_CATCH(device)
    return 0;
  }

  OIDN_API void oidnSetDeviceErrorFunction(OIDNDevice hDevice, OIDNErrorFunction func, void* userPtr)
  {
    Device* device = reinterpret_cast<Device*>(hDevice);
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      device->setErrorFunction(reinterpret_cast<ErrorFunction>(func), userPtr);
    OIDN_CATCH(device)
  }

  OIDN_API OIDNError oidnGetDeviceError(OIDNDevice hDevice, const char** outMessage)
  {
    Device* device = reinterpret_cast<Device*>(hDevice);
    OIDN_TRY
      return static_cast<OIDNError>(Device::getError(device, outMessage));
    OIDN_CATCH(device)
    if (outMessage) *outMessage = "";
    return OIDN_ERROR_UNKNOWN;
  }

  OIDN_API void oidnCommitDevice(OIDNDevice hDevice)
  {
    Device* device = reinterpret_cast<Device*>(hDevice);
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      device->commit();
    OIDN_CATCH(device)
  }

  OIDN_API void oidnSyncDevice(OIDNDevice hDevice)
  {
    Device* device = reinterpret_cast<Device*>(hDevice);
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      device->wait();
    OIDN_CATCH(device)
  }

  OIDN_API OIDNBuffer oidnNewBuffer(OIDNDevice hDevice, size_t byteSize)
  {
    Device* device = reinterpret_cast<Device*>(hDevice);
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      device->checkCommitted();
      Ref<Buffer> buffer = device->getEngine()->newBuffer(byteSize, Storage::Host);
      return reinterpret_cast<OIDNBuffer>(buffer.detach());
    OIDN_CATCH(device)
    return nullptr;
  }

  OIDN_API OIDNBuffer oidnNewBufferWithStorage(OIDNDevice hDevice, size_t byteSize, OIDNStorage storage)
  {
    Device* device = reinterpret_cast<Device*>(hDevice);
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      device->checkCommitted();
      Ref<Buffer> buffer = device->getEngine()->newBuffer(byteSize, static_cast<Storage>(storage));
      return reinterpret_cast<OIDNBuffer>(buffer.detach());
    OIDN_CATCH(device)
    return nullptr;
  }

  OIDN_API OIDNBuffer oidnNewSharedBuffer(OIDNDevice hDevice, void* devPtr, size_t byteSize)
  {
    Device* device = reinterpret_cast<Device*>(hDevice);
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      device->checkCommitted();
      Ref<Buffer> buffer = device->getEngine()->newBuffer(devPtr, byteSize);
      return reinterpret_cast<OIDNBuffer>(buffer.detach());
    OIDN_CATCH(device)
    return nullptr;
  }

  OIDN_API OIDNBuffer oidnNewSharedBufferFromFD(OIDNDevice hDevice,
                                                OIDNExternalMemoryTypeFlag fdType,
                                                int fd, size_t byteSize)
  {
    Device* device = reinterpret_cast<Device*>(hDevice);
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      device->checkCommitted();
      if (!(static_cast<ExternalMemoryTypeFlag>(fdType) & device->getExternalMemoryTypes()))
        throw Exception(Error::InvalidArgument, "external memory type not supported by the device");
      Ref<Buffer> buffer = device->getEngine()->newExternalBuffer(
        static_cast<ExternalMemoryTypeFlag>(fdType), fd, byteSize);
      return reinterpret_cast<OIDNBuffer>(buffer.detach());
    OIDN_CATCH(device)
    return nullptr;
  }

  OIDN_API OIDNBuffer oidnNewSharedBufferFromWin32Handle(OIDNDevice hDevice,
                                                         OIDNExternalMemoryTypeFlag handleType,
                                                         void* handle, const void* name, size_t byteSize)
  {
    Device* device = reinterpret_cast<Device*>(hDevice);
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      device->checkCommitted();
      if (!(static_cast<ExternalMemoryTypeFlag>(handleType) & device->getExternalMemoryTypes()))
        throw Exception(Error::InvalidArgument, "external memory type not supported by the device");
      if ((!handle && !name) || (handle && name))
        throw Exception(Error::InvalidArgument, "exactly one of the external memory handle and name must be non-null");
      Ref<Buffer> buffer = device->getEngine()->newExternalBuffer(
        static_cast<ExternalMemoryTypeFlag>(handleType), handle, name, byteSize);
      return reinterpret_cast<OIDNBuffer>(buffer.detach());
    OIDN_CATCH(device)
    return nullptr;
  }

  OIDN_API void oidnRetainBuffer(OIDNBuffer hBuffer)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(hBuffer);
    retainObject(buffer);
  }

  OIDN_API void oidnReleaseBuffer(OIDNBuffer hBuffer)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(hBuffer);
    releaseObject(buffer);
  }

  OIDN_API void* oidnMapBuffer(OIDNBuffer hBuffer, OIDNAccess access, size_t byteOffset, size_t byteSize)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(hBuffer);
    OIDN_TRY
      checkHandle(hBuffer);
      OIDN_LOCK(buffer);
      return buffer->map(byteOffset, byteSize, (Access)access);
    OIDN_CATCH(buffer)
    return nullptr;
  }

  OIDN_API void oidnUnmapBuffer(OIDNBuffer hBuffer, void* mappedPtr)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(hBuffer);
    OIDN_TRY
      checkHandle(hBuffer);
      OIDN_LOCK(buffer);
      return buffer->unmap(mappedPtr);
    OIDN_CATCH(buffer)
  }

  OIDN_API void oidnReadBuffer(OIDNBuffer hBuffer, size_t byteOffset, size_t byteSize, void* dstHostPtr)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(hBuffer);
    OIDN_TRY
      checkHandle(hBuffer);
      OIDN_LOCK(buffer);
      buffer->read(byteOffset, byteSize, dstHostPtr);
    OIDN_CATCH(buffer);
  }

  OIDN_API void oidnReadBufferAsync(OIDNBuffer hBuffer, size_t byteOffset, size_t byteSize, void* dstHostPtr)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(hBuffer);
    OIDN_TRY
      checkHandle(hBuffer);
      OIDN_LOCK(buffer);
      buffer->read(byteOffset, byteSize, dstHostPtr, SyncMode::Async);
    OIDN_CATCH(buffer);
  }

  OIDN_API void oidnWriteBuffer(OIDNBuffer hBuffer, size_t byteOffset, size_t byteSize, const void* srcHostPtr)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(hBuffer);
    OIDN_TRY
      checkHandle(hBuffer);
      OIDN_LOCK(buffer);
      buffer->write(byteOffset, byteSize, srcHostPtr);
    OIDN_CATCH(buffer);
  }

  OIDN_API void oidnWriteBufferAsync(OIDNBuffer hBuffer, size_t byteOffset, size_t byteSize, const void* srcHostPtr)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(hBuffer);
    OIDN_TRY
      checkHandle(hBuffer);
      OIDN_LOCK(buffer);
      buffer->write(byteOffset, byteSize, srcHostPtr, SyncMode::Async);
    OIDN_CATCH(buffer);
  }

  OIDN_API void* oidnGetBufferData(OIDNBuffer hBuffer)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(hBuffer);
    OIDN_TRY
      checkHandle(hBuffer);
      OIDN_LOCK(buffer);
      return buffer->getData();
    OIDN_CATCH(buffer)
    return nullptr;
  }

  OIDN_API size_t oidnGetBufferSize(OIDNBuffer hBuffer)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(hBuffer);
    OIDN_TRY
      checkHandle(hBuffer);
      OIDN_LOCK(buffer);
      return buffer->getByteSize();
    OIDN_CATCH(buffer)
    return 0;
  }

  OIDN_API OIDNFilter oidnNewFilter(OIDNDevice hDevice, const char* type)
  {
    Device* device = reinterpret_cast<Device*>(hDevice);
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      device->checkCommitted();
      Ref<Filter> filter = device->newFilter(type);
      return reinterpret_cast<OIDNFilter>(filter.detach());
    OIDN_CATCH(device)
    return nullptr;
  }

  OIDN_API void oidnRetainFilter(OIDNFilter hFilter)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    retainObject(filter);
  }

  OIDN_API void oidnReleaseFilter(OIDNFilter hFilter)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    releaseObject(filter);
  }

  OIDN_API void oidnSetFilterImage(OIDNFilter hFilter, const char* name,
                                   OIDNBuffer hBuffer, OIDNFormat format,
                                   size_t width, size_t height,
                                   size_t byteOffset,
                                   size_t pixelByteStride, size_t rowByteStride)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    OIDN_TRY
      checkHandle(hFilter);
      checkHandle(hBuffer);
      OIDN_LOCK(filter);
      Ref<Buffer> buffer = reinterpret_cast<Buffer*>(hBuffer);
      if (buffer->getDevice() != filter->getDevice())
        throw Exception(Error::InvalidArgument, "the specified objects are bound to different devices");
      auto image = std::make_shared<Image>(buffer, static_cast<Format>(format),
                                           static_cast<int>(width), static_cast<int>(height),
                                           byteOffset, pixelByteStride, rowByteStride);
      filter->setImage(name, image);
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnSetSharedFilterImage(OIDNFilter hFilter, const char* name,
                                         void* devPtr, OIDNFormat format,
                                         size_t width, size_t height,
                                         size_t byteOffset,
                                         size_t pixelByteStride, size_t rowByteStride)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      auto image = std::make_shared<Image>(devPtr, static_cast<Format>(format),
                                           static_cast<int>(width), static_cast<int>(height),
                                           byteOffset, pixelByteStride, rowByteStride);
      filter->setImage(name, image);
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnRemoveFilterImage(OIDNFilter hFilter, const char* name)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      filter->removeImage(name);
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnSetSharedFilterData(OIDNFilter hFilter, const char* name,
                                        void* hostPtr, size_t byteSize)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      Data data(hostPtr, byteSize);
      filter->setData(name, data);
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnUpdateFilterData(OIDNFilter hFilter, const char* name)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      filter->updateData(name);
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnRemoveFilterData(OIDNFilter hFilter, const char* name)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      filter->removeData(name);
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnSetFilter1b(OIDNFilter hFilter, const char* name, bool value)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      filter->set1i(name, int(value));
    OIDN_CATCH(filter)
  }

  OIDN_API bool oidnGetFilter1b(OIDNFilter hFilter, const char* name)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      return filter->get1i(name);
    OIDN_CATCH(filter)
    return false;
  }

  OIDN_API void oidnSetFilter1i(OIDNFilter hFilter, const char* name, int value)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      filter->set1i(name, value);
    OIDN_CATCH(filter)
  }

  OIDN_API int oidnGetFilter1i(OIDNFilter hFilter, const char* name)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      return filter->get1i(name);
    OIDN_CATCH(filter)
    return 0;
  }

  OIDN_API void oidnSetFilter1f(OIDNFilter hFilter, const char* name, float value)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      filter->set1f(name, value);
    OIDN_CATCH(filter)
  }

  OIDN_API float oidnGetFilter1f(OIDNFilter hFilter, const char* name)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      return filter->get1f(name);
    OIDN_CATCH(filter)
    return 0;
  }

  OIDN_API void oidnSetFilterProgressMonitorFunction(OIDNFilter hFilter, OIDNProgressMonitorFunction func, void* userPtr)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      filter->setProgressMonitorFunction(func, userPtr);
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnCommitFilter(OIDNFilter hFilter)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      filter->commit();
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnExecuteFilter(OIDNFilter hFilter)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      filter->execute();
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnExecuteFilterAsync(OIDNFilter hFilter)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      filter->execute(SyncMode::Async);
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnExecuteSYCLFilterAsync(OIDNFilter hFilter,
                                           const sycl::event* depEvents,
                                           int numDepEvents,
                                           sycl::event* doneEvent)
  {
    Filter* filter = reinterpret_cast<Filter*>(hFilter);
    OIDN_TRY
      // Check the parameters
      checkHandle(hFilter);
      if (numDepEvents < 0)
        throw Exception(Error::InvalidArgument, "invalid number of dependent events");

      OIDN_LOCK(filter);

      // Check whether the filter belongs to a SYCL device
      if (filter->getDevice()->getType() != DeviceType::SYCL)
        throw Exception(Error::InvalidOperation, "filter does not belong to a SYCL device");
      SYCLDeviceBase* device = static_cast<SYCLDeviceBase*>(filter->getDevice());

      // Execute the filter
      device->setDepEvents(depEvents, numDepEvents);
      filter->execute(SyncMode::Async);

      // Output the completion event (optional)
      if (doneEvent != nullptr)
        device->getDoneEvent(*doneEvent);
    OIDN_CATCH(filter)
  }

OIDN_API_NAMESPACE_END
