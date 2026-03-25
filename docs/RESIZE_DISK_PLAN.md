# Disk Resize Plan: Give Asahi More Space

## Current State (932GB drive)
```
p1   500M   Apple Silicon boot
p2   750.3G macOS APFS          <-- WAY too big
p3   2.3G   Apple APFS stub (Asahi)
p4   500M   EFI (/boot/efi)
p5   1G     Linux /boot
p6   172G   Linux / and /home   <-- only 32G free
p7   5G     Apple Silicon recovery
```

## Target State
- macOS: ~80G (plenty for a minimal macOS + recovery tools)
- New Linux partition: ~670G (for /repo, models, data)
- Asahi root (p6): stays 172G (OS + packages)

## Phase 1: Shrink macOS (FROM macOS)

Boot into macOS, open Terminal, then:

```bash
# Check current APFS usage first
diskutil apfs list

# Shrink the macOS APFS container to 80GB
# (adjust if you need more - 80G is generous for a minimal macOS)
diskutil apfs resizeContainer disk0s2 80g
```

If it complains about disk0s2, check the actual device name with `diskutil list` first.
The APFS container name might be slightly different.

**This will take a while.** It moves data and shrinks the container from the end.
If it says there's not enough free space, delete stuff from macOS first (caches, downloads, etc).

## Phase 2: Create Linux Partition (FROM Asahi Linux)

Boot back into Asahi. The free space is between the shrunken p2 and p3.

```bash
# Verify the free space exists - look for the gap between p2 and p3
sudo fdisk -l /dev/nvme0n1

# Use fdisk to create a new partition in the gap
sudo fdisk /dev/nvme0n1
```

In fdisk:
```
Command: n            (new partition)
Partition number: 8   (next available)
First sector: [accept default - should be right after shrunken p2]
Last sector: [accept default or type the sector just before p3 starts: 196826629]
Type: w               (write and exit)
```

**IMPORTANT**: p3 starts at sector 196826630. Make sure your new partition ends at or before 196826629.

Then format and mount:
```bash
# Format as ext4
sudo mkfs.ext4 -L data /dev/nvme0n1p8

# Create mount point
sudo mkdir -p /data

# Add to fstab for permanent mount
echo 'LABEL=data /data ext4 defaults 0 2' | sudo tee -a /etc/fstab

# Mount it
sudo mount /data

# Set ownership
sudo chown z:z /data
```

## Phase 3: Symlinks for Models/Repos

```bash
# Use /data for the big stuff
mkdir -p /data/models /data/repo /data/slowrepo /data/vmrepo

# Symlink from home if convenient
ln -s /data/models ~/models
ln -s /data/repo ~/repo
```

## Notes
- macOS APFS shrink is reversible — you can grow it back later
- The Linux partition is a new p8, not extending p6 — this avoids any risk to the running OS
- When /repo /slowrepo /vmrepo are mounted later, you'll have even more space
- 80G macOS is enough for: macOS itself (~15G), Xcode CLI tools, recovery, firmware updates
