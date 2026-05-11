param(
    [Parameter(Mandatory=$true)]
    [string]$ModelName
)

$modelDir = "models"
$modelUrl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/$ModelName.pkl"
$modelPath = "$modelDir\$ModelName.pkl"

New-Item -ItemType Directory -Force -Path $modelDir
curl.exe -L -o $modelPath $modelUrl