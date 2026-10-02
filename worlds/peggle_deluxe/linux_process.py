from typing import List, Optional, Tuple

import os
import struct


# Attaches to the game running under Wine / Proton through /proc. Mirrors the subset of the Pymem API used by
# GameStateManager. Writes through /proc/<pid>/mem ignore page protections, so code can be patched directly.
class LinuxProcess:
    # Arena header stored inside the game process so a reconnecting client never hands out memory that hooks
    # installed by a previous client still point to
    arena_magic: bytes = b"MWGG"
    arena_header_size: int = 16
    arena_alignment: int = 16

    pid: int
    base_address: int

    def __init__(self, pid: int, module_name: str) -> None:
        self.pid = pid
        self.base_address = self._find_base_address(module_name)
        self._fd: Optional[int] = os.open(f"/proc/{pid}/mem", os.O_RDWR)
        self._arena: Optional[Tuple[int, int]] = None

    @staticmethod
    def list_pids(process_name: str) -> List[int]:
        pids: List[int] = list()

        entry: str
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue

            try:
                with open(f"/proc/{entry}/comm") as comm_file:
                    comm: str = comm_file.read().strip()
            except OSError:
                continue

            if process_name.lower() in comm.lower():
                pids.append(int(entry))

        return pids

    def close(self) -> bool:
        if self._fd is None:
            return False

        os.close(self._fd)
        self._fd = None

        return True

    def read_bytes(self, address: int, length: int) -> bytes:
        data: bytes = os.pread(self._fd, length, address)

        if len(data) != length:
            raise OSError(f"Short read at 0x{address:x}")

        return data

    def write_bytes(self, address: int, data: bytes, length: int) -> None:
        if os.pwrite(self._fd, data[:length], address) != length:
            raise OSError(f"Short write at 0x{address:x}")

    def read_int(self, address: int) -> int:
        return struct.unpack("<i", self.read_bytes(address, 4))[0]

    def read_uint(self, address: int) -> int:
        return struct.unpack("<I", self.read_bytes(address, 4))[0]

    def write_int(self, address: int, value: int) -> None:
        self.write_bytes(address, struct.pack("<i", value), 4)

    def read_string(self, address: int, length: int) -> str:
        return self.read_bytes(address, length).split(b"\x00", 1)[0].decode("utf-8", errors="replace")

    # There is no way to call VirtualAllocEx from outside of Wine, so allocations come from the zeroed slack between
    # the end of a writable section and its page boundary. The game never touches that memory.
    def allocate(self, size: int) -> int:
        start: int
        end: int
        start, end = self._get_arena()

        used: int = struct.unpack("<I", self.read_bytes(start + 4, 4))[0]
        address: int = start + self.arena_header_size + used

        if address + size > end:
            raise MemoryError("Out of code cave space")

        used += (size + self.arena_alignment - 1) // self.arena_alignment * self.arena_alignment
        self.write_bytes(start + 4, struct.pack("<I", used), 4)

        return address

    def _find_base_address(self, module_name: str) -> int:
        with open(f"/proc/{self.pid}/maps") as maps_file:
            line: str
            for line in maps_file:
                fields: List[str] = line.split(maxsplit=5)

                if len(fields) < 6 or int(fields[2], 16) != 0:
                    continue

                if module_name.lower() in os.path.basename(fields[5].strip()).lower():
                    return int(fields[0].split("-")[0], 16)

        raise OSError(f"{module_name} is not mapped in process {self.pid}")

    def _get_arena(self) -> Tuple[int, int]:
        if self._arena is not None:
            return self._arena

        pe_offset: int = self.read_uint(self.base_address + 0x3C)
        header: bytes = self.read_bytes(self.base_address + pe_offset, 0x18)

        section_count: int = struct.unpack_from("<H", header, 0x6)[0]
        optional_header_size: int = struct.unpack_from("<H", header, 0x14)[0]
        section_table: bytes = self.read_bytes(
            self.base_address + pe_offset + 0x18 + optional_header_size, section_count * 40
        )

        best: Optional[Tuple[int, int]] = None

        i: int
        for i in range(section_count):
            virtual_size: int
            virtual_address: int
            virtual_size, virtual_address = struct.unpack_from("<II", section_table, i * 40 + 0x8)
            characteristics: int = struct.unpack_from("<I", section_table, i * 40 + 0x24)[0]

            if not characteristics & 0x80000000:  # IMAGE_SCN_MEM_WRITE
                continue

            start: int = self.base_address + virtual_address + virtual_size
            start = (start + self.arena_alignment - 1) // self.arena_alignment * self.arena_alignment
            end: int = (start + 0xFFF) & ~0xFFF

            if best is None or end - start > best[1] - best[0]:
                best = (start, end)

        if best is None or best[1] - best[0] <= self.arena_header_size:
            raise MemoryError("No section slack available for code caves")

        if not self._is_executable(best[0]):
            raise MemoryError("Section slack is not executable")

        magic: bytes = self.read_bytes(best[0], 4)

        if magic != self.arena_magic:
            if any(self.read_bytes(best[0], best[1] - best[0])):
                raise MemoryError("Section slack is not empty")

            self.write_bytes(best[0], self.arena_magic + bytes(4), 8)

        self._arena = best

        return best

    def _is_executable(self, address: int) -> bool:
        with open(f"/proc/{self.pid}/maps") as maps_file:
            line: str
            for line in maps_file:
                fields: List[str] = line.split()
                start, end = (int(value, 16) for value in fields[0].split("-"))

                if start <= address < end:
                    return "x" in fields[1]

        return False
