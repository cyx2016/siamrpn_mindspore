#include "MxBase/Log/Log.h"
#include "SiamRPN.h"

namespace {
const uint32_t DEVICE_ID = 0;
}  // namespace

int main(int argc, char *argv[]) {
  if (argc <= 3) {
    LogWarn << "Please input image path, such as './build/siamRPN_mindspore "
               "[om_file_path] [dataset_path] [dataset_name]'.";
    return APP_ERR_OK;
  }

  InitParam initParam = {};
  initParam.deviceId = DEVICE_ID;
  initParam.checkTensor = false;
  initParam.modelPath = argv[1];

  auto inferSiamRPN = std::make_shared<SiamRPN>();
  APP_ERROR ret = inferSiamRPN->Init(initParam);
  if (ret != APP_ERR_OK) {
    LogError << "SiamRPN init failed, ret=" << ret << ".";
    return ret;
  }
  std::string dataset_path = argv[2];
  std::string dataset_name = argv[3];
  ret = inferSiamRPN->Process(dataset_path, dataset_name);
  if (ret != APP_ERR_OK) {
    LogError << "SiamRPN process failed, ret=" << ret << ".";
    inferSiamRPN->DeInit();
    return ret;
  }
  inferSiamRPN->DeInit();
  return APP_ERR_OK;
}
