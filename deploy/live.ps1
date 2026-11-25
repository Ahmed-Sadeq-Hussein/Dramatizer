# deploy/live.ps1 | blue_green.py with the live site's settings (Milestone 4, step 5).
#
# The live site runs in Docker: slots on 5106 (blue) and 5107 (green), behind the front door on 5005 (playit, the
# pinned clients) and 5105 (the Cloudflare tunnel to https://app.dramatizer.dev). Its state is the named volume
# dramatizer-state (moved there once with deploy/move_state.py): do not start `main_backend.py api` on the host
# any more, or the two drift apart.
#
#   .\deploy\site.ps1 up          # the live slot + the front door and tunnel containers (docker/site.compose.yml)
#   .\deploy\live.ps1 release <image id> --release reports/milestone4/05-release/records/v5.json
#   .\deploy\live.ps1 rollback
#   .\deploy\live.ps1 status
#
# Which slot is live: history/live/front_door.json (the front door container sees only history/live/).
#
# --free-gpu: with the chatbot on, one 10 GB GPU holds one version, so a release or rollback pauses the site for
# about 2 minutes (the live version is restarted if the candidate is blocked).
$py = if ($env:DRAMATIZER_PYTHON) { $env:DRAMATIZER_PYTHON } else { "E:\envs\dramatizer\python.exe" }
$root = Split-Path -Parent $PSScriptRoot
# A release starts with deploy/container.env and then deploy/live.env (VOICES_DIR=/team-voices): make its record with
# the same two files, `python deploy/release.py v6 ... --env deploy/container.env --env deploy/live.env`
$extra = @()
if ($args.Count -gt 0 -and $args[0] -eq "release") {
    $extra = @("--env", "$root\deploy\container.env", "--env", "$root\deploy\live.env")
}
& $py "$root\deploy\blue_green.py" `
    --free-gpu --volume dramatizer-state `
    --voice-pack "D:/Dramatizer voice pack/46a182c41bc13df2754560273a9224d53b6f8e26" `
    --team-voices "D:/voice data/Voices" `
    --fingerprint b00a5ab5bf49bd120df49c1122a967f0e2e775c08f4f6507affcc343d53e4791 `
    --password-file "D:/Dramatizer secrets/smoke-test.txt" @args @extra
exit $LASTEXITCODE
