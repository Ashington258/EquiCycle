"""CH100 HI91 binary protocol parser extracted from project history."""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from typing import Any

CHSYNC1 = 0x5A
CHSYNC2 = 0xA5
CH_HDR_SIZE = 6
FRAME_TAG_HI91 = 0x91
HI91_PAYLOAD_SIZE = 76
MAX_PAYLOAD_SIZE = 4096


@dataclass
class CH100Frame:
    frame_type: int
    sync_time: float
    temperature: int
    pressure: float
    system_time_ms: int
    acc: tuple[float, float, float]
    gyr: tuple[float, float, float]
    mag: tuple[float, float, float]
    roll: float
    pitch: float
    yaw: float
    quat: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CH100Parser:
    """Incrementally parse CH100 frames from arbitrary serial chunks."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.valid_frames = 0
        self.crc_errors = 0
        self.invalid_lengths = 0
        self.skipped_bytes = 0

    @staticmethod
    def crc16_update(crc: int, data: bytes | bytearray) -> int:
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                temp = crc << 1
                if crc & 0x8000:
                    temp ^= 0x1021
                crc = temp
        return crc & 0xFFFF

    @staticmethod
    def _parse_hi91(payload: bytes | bytearray) -> CH100Frame:
        if len(payload) < HI91_PAYLOAD_SIZE:
            raise ValueError(
                f"HI91 payload is too short: {len(payload)} < {HI91_PAYLOAD_SIZE}"
            )

        return CH100Frame(
            frame_type=payload[0],
            sync_time=struct.unpack_from("<H", payload, 1)[0] * 1e-3,
            temperature=struct.unpack_from("<b", payload, 3)[0],
            pressure=struct.unpack_from("<f", payload, 4)[0],
            system_time_ms=struct.unpack_from("<I", payload, 8)[0],
            acc=struct.unpack_from("<3f", payload, 12),
            gyr=struct.unpack_from("<3f", payload, 24),
            mag=struct.unpack_from("<3f", payload, 36),
            roll=struct.unpack_from("<f", payload, 48)[0],
            pitch=struct.unpack_from("<f", payload, 52)[0],
            yaw=struct.unpack_from("<f", payload, 56)[0],
            quat=struct.unpack_from("<4f", payload, 60),
        )

    def parse(self, new_data: bytes | bytearray) -> list[dict[str, Any]]:
        self.buffer.extend(new_data)
        frames: list[dict[str, Any]] = []

        while len(self.buffer) >= CH_HDR_SIZE:
            if self.buffer[0] != CHSYNC1 or self.buffer[1] != CHSYNC2:
                del self.buffer[0]
                self.skipped_bytes += 1
                continue

            payload_length = struct.unpack_from("<H", self.buffer, 2)[0]
            if payload_length == 0 or payload_length > MAX_PAYLOAD_SIZE:
                del self.buffer[0]
                self.invalid_lengths += 1
                continue

            total_length = CH_HDR_SIZE + payload_length
            if len(self.buffer) < total_length:
                break

            frame_data = bytes(self.buffer[:total_length])
            del self.buffer[:total_length]

            crc_received = struct.unpack_from("<H", frame_data, 4)[0]
            crc_calculated = self.crc16_update(
                0, frame_data[:4] + frame_data[CH_HDR_SIZE:]
            )
            if crc_calculated != crc_received:
                self.crc_errors += 1
                continue

            payload = frame_data[CH_HDR_SIZE:]
            if payload[0] != FRAME_TAG_HI91:
                continue

            try:
                frames.append(self._parse_hi91(payload).to_dict())
                self.valid_frames += 1
            except (ValueError, struct.error):
                self.invalid_lengths += 1

        return frames
