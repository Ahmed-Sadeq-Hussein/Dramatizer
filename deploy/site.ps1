# deploy/site.ps1 | Start, stop or check the whole live site in Docker.
#
#   .\deploy\site.ps1 up       # the live slot, the front door (5005, 5105) and the tunnel (https://app.dramatizer.dev)
#   .\deploy\site.ps1 down     # stop all three (nothing is deleted)
#   .\deploy\site.ps1 status   # what runs, and whether the site answers
#
# Docker Desktop must be running. Releases and rollbacks are still .\deploy\live.ps1 release|rollback.
# Not "Stop": Windows PowerShell 5.1 treats any text a native command writes to stderr as an error. Exit codes are
# checked where they matter, and `throw` still stops the script.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$compose = "$root\docker\site.compose.yml"
$state = "$root\history\live\front_door.json"
$cf = Join-Path $env:USERPROFILE ".cloudflared"
$command = if ($args.Count -gt 0) { $args[0] } else { "status" }

function Get-Live {
    if (-not (Test-Path $state)) { return $null }
    return (Get-Content $state -Raw | ConvertFrom-Json).live
}

function Set-TunnelConfig {
    # A copy of the tunnel's config.yml for the container: the same tunnel and hostname, but it reaches the front
    # door by its container name. No secret in it; the credentials file stays where it is and is mounted read-only.
    $config = Get-Content (Join-Path $cf "config.yml") -Raw
    $id = [regex]::Match($config, "(?m)^tunnel:\s*(\S+)").Groups[1].Value
    $hostname = [regex]::Match($config, "(?m)hostname:\s*(\S+)").Groups[1].Value
    if (-not $id -or -not $hostname) { throw "No tunnel in $cf\config.yml: run python main_backend.py tunnel --setup first." }
    @"
# Written by deploy/site.ps1 for the container dramatizer-tunnel (docker/site.compose.yml).
tunnel: $id
credentials-file: /etc/cloudflared/credentials.json
ingress:
  - hostname: $hostname
    service: https://dramatizer-front-door:5105
    originRequest:
      noTLSVerify: true        # the API's own certificate is self-signed; this hop stays inside Docker
  - service: http_status:404
"@ | Set-Content -Encoding ascii (Join-Path $cf "docker.yml")
    $env:TUNNEL_ID = $id
    return $hostname
}

function Test-Site($url) {
    try {
        $out = & curl.exe -sk -m 5 $url
        return $out
    } catch { return $null }
}

switch ($command) {
    "up" {
        docker network inspect dramatizer *> $null
        if ($LASTEXITCODE -ne 0) { docker network create dramatizer | Out-Null }
        $live = Get-Live
        if (-not $live) { throw "Nothing is live in $state yet: release a version first with .\deploy\live.ps1 release." }
        # Slots made before the network existed: join it, and come back after a reboot like the new ones do
        foreach ($slot in @("dramatizer-blue", "dramatizer-green")) {
            docker container inspect $slot *> $null
            if ($LASTEXITCODE -ne 0) { continue }
            $nets = docker inspect -f "{{json .NetworkSettings.Networks}}" $slot
            if ($nets -notmatch '"dramatizer"') { docker network connect dramatizer $slot }
            docker update --restart unless-stopped $slot | Out-Null
        }
        Write-Host "Starting the live slot: $($live.container) (release $($live.release))"
        docker start $live.container | Out-Null
        $hostname = Set-TunnelConfig
        docker compose -f $compose up -d
        if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }
        Write-Host "Waiting for the site to be ready (about 75 s after a cold start)..."
        $deadline = (Get-Date).AddMinutes(4)
        while ((Get-Date) -lt $deadline) {
            $ready = Test-Site "https://127.0.0.1:5005/api/v1/ready"
            if ($ready -match '"ready":\s*true') { break }
            Start-Sleep -Seconds 3
        }
        Write-Host "ready: $ready"
        Write-Host "Open https://$hostname (or https://127.0.0.1:5005 on this PC)."
    }
    "down" {
        docker compose -f $compose down
        $live = Get-Live
        if ($live) { docker stop $live.container | Out-Null; Write-Host "Stopped $($live.container)." }
    }
    "status" {
        docker ps -a --filter "name=dramatizer-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        $live = Get-Live
        if ($live) { Write-Host "Live: $($live.container) (release $($live.release))" }
        Write-Host "https://127.0.0.1:5005/api/v1/ready -> $(Test-Site 'https://127.0.0.1:5005/api/v1/ready')"
    }
    default { throw "Use: .\deploy\site.ps1 up | down | status" }
}
