# deploy/staging.ps1 | blue_green.py with the staging pair's settings: its own state volume, ports 5106/5107 behind
# the staging front door on 5108, renders in "D:/Dramatizer 2 renders/staging". The live site is not touched.
#
#   python deploy/front_door.py --listen 5108 --state history/staging/front_door.json     # in its own window
#   .\deploy\staging.ps1 release <image id> --release reports/milestone4/05-release/records/v2.json
#   .\deploy\staging.ps1 rollback
#   .\deploy\staging.ps1 status
#
# DRAMATIZER_PYTHON picks the Python environment (default E:\envs\dramatizer). The fingerprint is the staging
# certificate's, in the staging state volume.
$py = if ($env:DRAMATIZER_PYTHON) { $env:DRAMATIZER_PYTHON } else { "E:\envs\dramatizer\python.exe" }
$root = Split-Path -Parent $PSScriptRoot
& $py "$root\deploy\blue_green.py" `
    --prefix staging --state "$root\history\staging\front_door.json" `
    --volume dramatizer-state-staging --output "D:/Dramatizer 2 renders/staging" `
    --voice-pack "D:/Dramatizer voice pack/46a182c41bc13df2754560273a9224d53b6f8e26" `
    --secrets-env "" `
    --fingerprint fa9ba7410eece32a4c7cebc5a800a2a9f4b91bf4f9b0fc451dd156026a66aa37 `
    --password-file "D:/Dramatizer secrets/staging/smoke-test.txt" @args
exit $LASTEXITCODE
