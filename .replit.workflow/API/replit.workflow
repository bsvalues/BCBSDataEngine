{
  "type": "web",
  "command": "./start_api.sh",
  "onBoot": "./start_api.sh",
  "restartOn": {
    "files": ["./api.py", "./run_api.py", "./start_api.sh"]
  },
  "deployEnv": {
    "AUTO_START": "true",
    "HIDE_URL": "false"
  }
}