# Typically it's located in ~/Android/Sdk/emulator/emulator or
# ~/Library/Android/sdk/emulator/emulator
EMULATOR_NAME=AndroidWorldAvd # From previous step
~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -no-snapshot -grpc 8554