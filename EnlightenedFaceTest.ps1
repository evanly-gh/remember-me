$base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("EnlightenedFace.png"))
"{""image"": ""$base64""}" | Out-File -Encoding utf8 payload.json
curl.exe -X POST http://localhost:7860/analyze-base64 `
  -H "Content-Type: application/json" `
  -d "@payload.json"