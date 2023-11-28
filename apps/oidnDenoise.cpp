// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common/common.h"
#include "common/timer.h"
#include "utils/arg_parser.h"
#include "utils/image_io.h"
#include "utils/device_info.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <limits>
#include <cmath>
#include <signal.h>
#ifdef VTUNE
#include <ittnotify.h>
#endif

#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <filesystem>
namespace fs = std::filesystem;

static std::string replaceSubstring(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if (start_pos != std::string::npos) {
        str.replace(start_pos, from.length(), to);
    }
    return str;
}

OIDN_NAMESPACE_USING

std::shared_ptr<ImageBuffer> loadImageEXR(const DeviceRef& device,
                                          const std::string& filename)
{
  int C = 4;
  int H, W;
  float* rgba;
  int ret = LoadEXR(&rgba, &W, &H, filename.c_str(), nullptr);
  // Read the pixels
  auto image = std::make_shared<ImageBuffer>(device, W, H, 3, DataType::Float32);

  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      for (int c = 0; c < 3; ++c) {
        int i = ((h * W) + w) * C + c; 
        int ti = ((h * W) + w) * 3 + c; 
        image->set(ti, rgba[i]);
      }
    }
  }

  return image;
}

std::shared_ptr<ImageBuffer> loadImageJPG(const DeviceRef& device,
                                          const std::string& filename)
{
  int C, H, W;
  float* data = stbi_loadf(filename.c_str(), &W, &H, &C, 0);
  if (C != 3) {
    puts(filename.c_str());
    puts("C != 3");
    exit(-1);
  }
  // Read the pixels
  auto image = std::make_shared<ImageBuffer>(device, W, H, 3, DataType::Float32);

  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      for (int c = 0; c < C; ++c) {
        int i = ((h * W) + w) * C + c; 
        image->set(i, data[i]);
      }
    }
  }
  stbi_image_free(data);
  return image;
}
void printUsage()
{
  std::cout << "Intel(R) Open Image Denoise" << std::endl;
  std::cout << "usage: oidnDenoise [-d/--device [0-9]+|default|cpu|sycl|cuda|hip]" << std::endl
            << "                   [-f/--filter RT|RTLightmap]" << std::endl
            << "                   [--hdr color.pfm] [--ldr color.pfm] [--srgb] [--dir directional.pfm]" << std::endl
            << "                   [--alb albedo.pfm] [--nrm normal.pfm] [--clean_aux]" << std::endl
            << "                   [--is/--input_scale value]" << std::endl
            << "                   [-o/--output output.pfm] [-r/--ref reference_output.pfm]" << std::endl
            << "                   [-t/--type float|half]" << std::endl
            << "                   [-q/--quality default|h|high|b|balanced]" << std::endl
            << "                   [-w/--weights weights.tza]" << std::endl
            << "                   [--threads n] [--affinity 0|1] [--maxmem MB] [--inplace]" << std::endl
            << "                   [-n times_to_run] [-v/--verbose 0-3]" << std::endl
            << "                   [--ld|--list_devices] [-h/--help]" << std::endl;
}

void errorCallback(void* userPtr, Error error, const char* message)
{
  throw std::runtime_error(message);
}

volatile bool isCancelled = false;

void signalHandler(int signal)
{
  isCancelled = true;
}

bool progressCallback(void* userPtr, double n)
{
  if (isCancelled)
  {
    std::cout << std::endl;
    return false;
  }
  std::cout << "\rDenoising " << int(n * 100.) << "%" << std::flush;
  return true;
}

std::vector<char> loadFile(const std::string& filename)
{
  std::ifstream file(filename, std::ios::binary);
  if (file.fail())
    throw std::runtime_error("cannot open file: '" + filename + "'");
  file.seekg(0, file.end);
  const size_t size = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  if (file.fail())
    throw std::runtime_error("error reading from file: '" + filename + "'");
  return buffer;
}

int main(int argc, char* argv[])
{
  DeviceType deviceType = DeviceType::Default;
  PhysicalDeviceRef physicalDevice;
  std::string filterType = "RT";
  std::string colorFilename, albedoFilename, normalFilename;
  std::string outputFilename, refFilename;
  std::string weightsFilename;
  Quality quality = Quality::Default;
  bool hdr = false;
  bool srgb = false;
  bool directional = false;
  float inputScale = std::numeric_limits<float>::quiet_NaN();
  bool cleanAux = false;
  DataType dataType = DataType::Void;
  int numRuns = 1;
  int numThreads = -1;
  int setAffinity = -1;
  int maxMemoryMB = -1;
  bool inplace = false;
  int verbose = -1;

  // Parse the arguments
  if (argc == 1)
  {
    printUsage();
    return 1;
  }

  try
  {
    ArgParser args(argc, argv);
    while (args.hasNext())
    {
      std::string opt = args.getNextOpt();
      if (opt == "d" || opt == "dev" || opt == "device")
      {
        std::string value = args.getNext();
        if (isdigit(value[0]))
          physicalDevice = fromString<int>(value);
        else
          deviceType = fromString<DeviceType>(value);
      }
      else if (opt == "f" || opt == "filter")
        filterType = args.getNextValue();
      else if (opt == "hdr")
      {
        colorFilename = args.getNextValue();
        hdr = true;
      }
      else if (opt == "ldr")
      {
        colorFilename = args.getNextValue();
        hdr = false;
      }
      else if (opt == "srgb")
        srgb = true;
      else if (opt == "dir")
      {
        colorFilename = args.getNextValue();
        directional = true;
      }
      else if (opt == "alb" || opt == "albedo")
        albedoFilename = args.getNextValue();
      else if (opt == "nrm" || opt == "normal")
        normalFilename = args.getNextValue();
      else if (opt == "o" || opt == "out" || opt == "output")
        outputFilename = args.getNextValue();
      else if (opt == "r" || opt == "ref" || opt == "reference")
        refFilename = args.getNextValue();
      else if (opt == "is" || opt == "input_scale" || opt == "input-scale" || opt == "inputScale" || opt == "inputscale")
        inputScale = args.getNextValue<float>();
      else if (opt == "clean_aux" || opt == "clean-aux" || opt == "cleanAux" || opt == "cleanaux")
        cleanAux = true;
      else if (opt == "t" || opt == "type")
      {
        const auto val = toLower(args.getNextValue());
        if (val == "f" || val == "float" || val == "fp32")
          dataType = DataType::Float32;
        else if (val == "h" || val == "half" || val == "fp16")
          dataType = DataType::Float16;
        else
          throw std::runtime_error("invalid data type");
      }
      else if (opt == "q" || opt == "quality")
      {
        const auto val = toLower(args.getNextValue());
        if (val == "default")
          quality = Quality::Default;
        else if (val == "h" || val == "high")
          quality = Quality::High;
        else if (val == "b" || val == "balanced")
          quality = Quality::Balanced;
        else
          throw std::runtime_error("invalid filter quality mode");
      }
      else if (opt == "w" || opt == "weights")
        weightsFilename = args.getNextValue();
      else if (opt == "n")
        numRuns = std::max(args.getNextValue<int>(), 1);
      else if (opt == "threads")
        numThreads = args.getNextValue<int>();
      else if (opt == "affinity")
        setAffinity = args.getNextValue<int>();
      else if (opt == "maxmem" || opt == "maxMemoryMB")
        maxMemoryMB = args.getNextValue<int>();
      else if (opt == "inplace")
        inplace = true;
      else if (opt == "v" || opt == "verbose")
        verbose = args.getNextValue<int>();
      else if (opt == "ld" || opt == "list_devices" || opt == "list-devices" || opt == "listDevices" || opt == "listdevices")
        return printPhysicalDevices();
      else if (opt == "h" || opt == "help")
      {
        printUsage();
        return 1;
      }
      else
        throw std::invalid_argument("invalid argument '" + opt + "'");
    }

  #if defined(OIDN_ARCH_X64)
    // Set MXCSR flags
    if (!refFilename.empty())
    {
      // In reference mode we have to disable the FTZ and DAZ flags to get accurate results
      _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
      _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
    }
    else
    {
      // Enable the FTZ and DAZ flags to maximize performance
      _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
      _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    }
  #endif

  std::vector<std::string> filenames;
  {
    for (const auto& entry : fs::directory_iterator(colorFilename)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().filename().string().find(".noising.jpg") == std::string::npos) {
            continue;
        }
        filenames.push_back(entry.path().string());
    }
  }

    // Initialize the denoising device
    std::cout << "Initializing device" << std::endl;
    Timer timer;

    DeviceRef device;
    if (physicalDevice)
      device = physicalDevice.newDevice();
    else
      device = newDevice(deviceType);

    const char* errorMessage;
    if (device.getError(errorMessage) != Error::None)
      throw std::runtime_error(errorMessage);
    device.setErrorFunction(errorCallback);

    if (numThreads > 0)
      device.set("numThreads", numThreads);
    if (setAffinity >= 0)
      device.set("setAffinity", bool(setAffinity));
    if (verbose >= 0)
      device.set("verbose", verbose);
    device.commit();

    const double deviceInitTime = timer.query();

    deviceType = device.get<DeviceType>("type");
    const int versionMajor = device.get<int>("versionMajor");
    const int versionMinor = device.get<int>("versionMinor");
    const int versionPatch = device.get<int>("versionPatch");

    std::cout << "  device=" << deviceType
              << ", version=" << versionMajor << "." << versionMinor << "." << versionPatch
              << ", msec=" << (1000. * deviceInitTime) << std::endl;

    // Load the input image
    std::shared_ptr<ImageBuffer> input, ref;
    std::shared_ptr<ImageBuffer> color, albedo, normal;

    std::cout << "Loading input" << std::endl;

    // if (!albedoFilename.empty())
    //   input = albedo = loadImageEXR(device, albedoFilename);

    // if (!normalFilename.empty())
    //   input = normal = loadImageEXR(device, normalFilename);

    // if (!colorFilename.empty())
    //   input = color = loadImageEXR(device, colorFilename);
    colorFilename = filenames[0];
    input = color = loadImageJPG(device, colorFilename);

    const int width  = input->getW();
    const int height = input->getH();
    std::cout << "Resolution: " << width << "x" << height << std::endl;

    // Initialize the output image
    std::shared_ptr<ImageBuffer> output = std::make_shared<ImageBuffer>(device, width, height, input->getC(), input->getDataType());

    for (auto i = 0; i < filenames.size(); i++) {
    colorFilename = filenames[i];
    color = loadImageJPG(device, colorFilename);
    albedo = loadImageEXR(device, replaceSubstring(colorFilename, ".noising.jpg", ".albedo.exr"));
    normal = loadImageEXR(device, replaceSubstring(colorFilename, ".noising.jpg", ".normal.exr"));
    // Initialize the denoising filter
    std::cout << "Initializing filter" << std::endl;
    timer.reset();

    FilterRef filter = device.newFilter(filterType.c_str());

    if (color)
      filter.setImage("color", color->getBuffer(), color->getFormat(), color->getW(), color->getH());
    if (albedo)
      filter.setImage("albedo", albedo->getBuffer(), albedo->getFormat(), albedo->getW(), albedo->getH());
    if (normal)
      filter.setImage("normal", normal->getBuffer(), normal->getFormat(), normal->getW(), normal->getH());

    filter.setImage("output", output->getBuffer(), output->getFormat(), output->getW(), output->getH());

    if (hdr) {
      filter.set("hdr", true);
    }

    filter.set("quality", OIDN_QUALITY_HIGH);

    filter.commit();

    const double filterInitTime = timer.query();

    std::cout << "  filter=" << filterType
              << ", msec=" << (1000. * filterInitTime) << std::endl;

    // Denoise the image
    {
      timer.reset();

      filter.execute();

      const double denoiseTime = timer.query();
      std::cout << "  msec=" << (1000. * denoiseTime) << std::endl;
    }

    {
      // Save output image
      std::cout << "Saving output" << std::endl;
      // saveImage(outputFilename, *output, srgb);
      outputFilename = replaceSubstring(colorFilename, ".noising.jpg", "");

      int w = output->getW();
      int h = output->getH();
      int c = output->getC();
      std::vector<uint8_t> color;
      std::vector<float> pixels;
      color.reserve(w * h * c);
      for (auto j = 0; j < w * h * c; j++) {
        float x = output->get(j);
        pixels.push_back(x);
        int v = clamp(int(std::pow(x, 1.0f / 2.2f) * 255.99f), 0, 255);
        color.push_back(v);
      }
      //SaveEXR(pixels.data(), w, h, 3, 1, (outputFilename + ".denoise.exr").c_str(), nullptr);

      stbi_flip_vertically_on_write(0);
      stbi_write_jpg((outputFilename + ".denoise.jpg").c_str(), w, h, c, color.data(), 100);
    }
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
