# Serve Foley-Omni Preview on a Linux PC (LAN / IP access)

This guide explains how to run the preview webapp on a **Linux GPU PC** so guests or teammates can open it in a browser using the machine’s **IP address** (or a hostname), not only `localhost`.

For the full one-shot deploy (checkpoints + Docker), see also [DEPLOY.md](../DEPLOY.md).

---

## What you need

- Linux PC with an NVIDIA GPU (prefer **≥24GB VRAM** for real inference)
- [Docker](https://docs.docker.com/engine/install/) + Compose v2
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Network path from client machines to this PC (same LAN, VPN, or opened cloud firewall)

Verify GPU in Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## 1. Deploy the app on the PC

```bash
git clone <your-repo-url> Foley-Omni-API
cd Foley-Omni-API
chmod +x deploy.sh deploy.bash
./deploy.sh --no-logs
```

This will:

1. Create `.env` and guest/admin tokens (if still placeholders)
2. Download checkpoints into `./ckpts` (if missing)
3. Build and start the container (`docker compose up --build -d`)

The container listens on **`0.0.0.0:8000`** (all interfaces), mapped to host port **8000**.

Check locally first:

```bash
curl -s http://127.0.0.1:8000/api/health
```

When ready, `model_loaded` becomes `true` (GPU weight load can take several minutes).

---

## 2. Find the PC’s IP address

```bash
hostname -I
# or
ip -4 addr show
```

Example: `192.168.1.50`

Clients will use:

```text
http://192.168.1.50:8000/
```

---

## 3. Open the firewall (port 8000)

### ufw

```bash
sudo ufw allow 8000/tcp
sudo ufw status
```

### firewalld

```bash
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload
```

### Cloud VM / public IP

Also allow **TCP 8000** in the cloud security group / NSG / firewall panel (AWS, GCP, Azure, etc.).

---

## 4. Share access with users

| Item | What to share |
|------|----------------|
| URL | `http://<PC-IP>:8000/` |
| Guest token | `ACCESS_TOKEN` from `.env` (printed by `./deploy.sh`) |

Users:

1. Open the URL in a browser  
2. Paste the access token  
3. Upload a short video (≤10s) and generate  

**Keep `ADMIN_TOKEN` private** — it can clear the queue.

Read tokens from the server:

```bash
grep -E '^(ACCESS_TOKEN|ADMIN_TOKEN)=' .env
```

API example from another machine:

```bash
curl -s -H "X-API-Key: YOUR_ACCESS_TOKEN" http://192.168.1.50:8000/api/me
```

---

## 5. Security notes (important for guest previews)

The app uses a **shared preview token**, not full user accounts. Anyone who can reach port 8000 and knows (or guesses) the token can submit jobs and use the GPU.

Recommended setups:

| Setup | When to use |
|-------|-------------|
| **Same office LAN only** | Quick internal demos; firewall still recommended |
| **VPN / Tailscale / WireGuard** | Best for remote guests without exposing the box to the open internet |
| **SSH tunnel** | One-off: user runs `ssh -L 8000:127.0.0.1:8000 user@pc` then opens `http://localhost:8000` |
| **Nginx reverse proxy + HTTPS** | When you want a domain name and TLS (see below) |

Do **not** expose port 8000 to the whole internet without VPN, allowlisting, or similar controls.

Rotate tokens after a demo round by editing `.env` and restarting:

```bash
# edit ACCESS_TOKEN / ADMIN_TOKEN in .env
docker compose up -d
```

---

## 6. Optional: Nginx reverse proxy (domain or nicer URL)

Install Nginx on the same PC, then proxy to the app. Example HTTP config:

```nginx
server {
    listen 80;
    server_name foley.example.com;   # or leave as default_server / use the PC IP

    client_max_body_size 200M;       # video uploads

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 3600s;     # long generation / polling
        proxy_send_timeout 3600s;
    }
}
```

Enable the site, reload Nginx, open port **80** (and **443** if using HTTPS).  
For TLS, use Certbot / Let’s Encrypt on `foley.example.com`.

If Nginx terminates TLS and the app stays on localhost:8000, you can leave Docker bound as-is and **not** open 8000 publicly—only open 80/443.

---

## 7. Day-2 operations

```bash
# Status
docker compose ps
curl -s http://127.0.0.1:8000/api/health

# Logs
docker compose logs -f foley-omni

# Restart after .env or code changes
docker compose up --build -d

# Stop
docker compose down
```

Uploads and results live under `./storage` on the host. Checkpoints stay in `./ckpts`.

---

## 8. Troubleshooting

| Symptom | What to check |
|---------|----------------|
| Works on PC, not from phone/laptop | Firewall / cloud SG; wrong IP; clients on different VLAN/VPN |
| Connection refused | Container down: `docker compose ps`; port mapping `8000:8000` |
| Page loads, jobs fail / 503 | Wait for `model_loaded: true`; check GPU + `./ckpts`; logs |
| Out of memory | Need more VRAM; try `CPU_OFFLOAD=true` in `.env` (helps mid-range GPUs, not tiny ones) |
| Upload fails / large video | Raise `client_max_body_size` if using Nginx; trim video to ≤10s |

---

## Quick checklist

- [ ] `nvidia-smi` works inside Docker with `--gpus all`
- [ ] `./deploy.sh` completed; `./ckpts` present
- [ ] `curl http://127.0.0.1:8000/api/health` OK
- [ ] Firewall allows TCP **8000** (or only 80/443 via Nginx)
- [ ] Users get `http://<PC-IP>:8000/` + `ACCESS_TOKEN`
- [ ] Access limited to LAN/VPN when possible
