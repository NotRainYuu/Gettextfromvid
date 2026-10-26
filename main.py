import os
import re
import sys
import json
import shutil
import ctypes
import tempfile
import argparse
import subprocess
from typing import Optional, Dict, Any, Tuple
import requests
import ctranslate2
import yt_dlp
from faster_whisper import WhisperModel

try:
    import pyperclip
    HAS_CLIPBOARD = True
except ImportError:
    HAS_CLIPBOARD = False

_P1 = bytes.fromhex("646f7579696e2e636f6d").decode("ascii")
_P2 = bytes.fromhex("696573646f7579696e2e636f6d").decode("ascii")

USER_AGENT_DESKTOP = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

USER_AGENT_MOBILE = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
)


def extract_url(text: str) -> Optional[str]:
    text = text.strip()
    if os.path.isfile(text):
        return text
    matches = re.findall(r"https?://[^\s<>\"'{}|\\^`]+", text)
    if matches:
        return re.sub(r"[),;.?!'\"\s]+$", "", matches[0])
    return None


def resolve_redirects(url: str) -> str:
    if any(k in url for k in ["v." + _P1, "vt.tiktok.com", "t.co", "youtu.be"]):
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": USER_AGENT_MOBILE},
                allow_redirects=True,
                timeout=10,
                stream=True,
            )
            if resp.url:
                return resp.url
        except Exception:
            pass
    return url


def extract_item_id(url: str) -> Optional[str]:
    m = re.search(r"/(?:video|note)/([0-9]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"modal_id=([0-9]+)", url)
    if m:
        return m.group(1)
    return None


def convert_media_to_mp3(input_file: str, output_file: str) -> str:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-vn",
        "-acodec", "libmp3lame",
        "-ac", "1",
        "-ar", "16000",
        "-q:a", "4",
        output_file,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion error: {res.stderr}")
    return output_file


def fallback_stream_extract(item_id: str, output_dir: str) -> Tuple[str, str]:
    api_url = f"https://www.{_P2}/aweme/v1/web/aweme/detail/?aweme_id={item_id}"
    resp = requests.get(api_url, headers={"User-Agent": USER_AGENT_MOBILE}, timeout=12)
    data = resp.json()
    detail = data.get("aweme_detail")
    if not detail:
        raise ValueError("Could not resolve media metadata from endpoint")
    title = detail.get("desc", f"video_{item_id}").strip()
    safe_title = re.sub(r'[\\/*?:"<>|]', "_", title)[:50] or f"video_{item_id}"
    play_addr = detail.get("video", {}).get("play_addr", {}).get("url_list", [])
    if not play_addr:
        play_addr = detail.get("music", {}).get("play_url", {}).get("url_list", [])
    if not play_addr:
        raise ValueError("No media stream URL found")
    stream_url = play_addr[0].replace("playwm", "play")
    temp_media = os.path.join(output_dir, f"{item_id}_raw.mp4")
    with requests.get(stream_url, headers={"User-Agent": USER_AGENT_DESKTOP}, stream=True, timeout=25) as r:
        r.raise_for_status()
        with open(temp_media, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
    target_mp3 = os.path.join(output_dir, f"{safe_title}_{item_id}.mp3")
    convert_media_to_mp3(temp_media, target_mp3)
    if os.path.exists(temp_media):
        try:
            os.remove(temp_media)
        except Exception:
            pass
    return target_mp3, title


def download_audio(input_target: str, output_dir: str) -> Tuple[str, str]:
    extracted = extract_url(input_target)
    if not extracted:
        raise ValueError(f"No valid link or media file found in input: {input_target}")
    if os.path.isfile(extracted):
        name = os.path.splitext(os.path.basename(extracted))[0]
        out_file = os.path.join(output_dir, f"{name}.mp3")
        convert_media_to_mp3(extracted, out_file)
        return out_file, name
    target_url = resolve_redirects(extracted)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title).60s_%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "http_headers": {"User-Agent": USER_AGENT_DESKTOP},
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(target_url, download=True)
            title = info.get("title", "audio_content")
            files = [
                os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
                if f.endswith(".mp3")
            ]
            if files:
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return files[0], title
    except Exception as e:
        if _P1 in target_url:
            vid = extract_item_id(target_url)
            if vid:
                return fallback_stream_extract(vid, output_dir)
        raise RuntimeError(f"Audio download failed: {e}") from e
    raise RuntimeError("Extracted audio could not be located on disk")


def has_cublas() -> bool:
    if sys.platform != "win32":
        return True
    for dll in ["cublas64_12.dll", "cublas64_11.dll"]:
        try:
            ctypes.CDLL(dll)
            return True
        except Exception:
            pass
    return False


def get_optimal_device() -> Tuple[str, str]:
    try:
        if ctranslate2.get_cuda_device_count() > 0 and has_cublas():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def format_timestamp(seconds: float, srt: bool = False) -> str:
    ms = round(seconds * 1000.0)
    hrs = ms // 3_600_000
    ms -= hrs * 3_600_000
    mins = ms // 60_000
    ms -= mins * 60_000
    secs = ms // 1_000
    ms -= secs * 1_000
    if srt:
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"
    return f"{mins:02d}:{secs:02d}.{ms // 100:01d}"


def transcribe_audio(
    audio_path: str,
    model_size: str,
    device_opt: str,
    lang_opt: Optional[str] = None,
) -> Dict[str, Any]:
    dev, compute = get_optimal_device() if device_opt == "auto" else (device_opt, "float16" if device_opt == "cuda" else "int8")
    print(f"[*] Loading model [{model_size}] on {dev} ({compute})...")
    try:
        model = WhisperModel(model_size, device=dev, compute_type=compute)
    except Exception as e:
        if dev == "cuda":
            print(f"[!] CUDA failed ({e}), falling back to CPU (int8)...")
            dev, compute = "cpu", "int8"
            model = WhisperModel(model_size, device=dev, compute_type=compute)
        else:
            raise e
    print("[*] Transcribing spoken content from audio...")
    prompt = "The following is a clear transcription with standard punctuation."
    try:
        segs_gen, info = model.transcribe(
            audio_path,
            language=lang_opt if lang_opt and lang_opt != "auto" else None,
            initial_prompt=prompt,
            vad_filter=True,
            beam_size=5,
        )
        first_seg = None
        it = iter(segs_gen)
        try:
            first_seg = next(it)
        except StopIteration:
            pass
    except Exception as e:
        if dev == "cuda":
            print(f"[!] GPU runtime issue ({e}), switching to CPU...")
            dev, compute = "cpu", "int8"
            model = WhisperModel(model_size, device=dev, compute_type=compute)
            segs_gen, info = model.transcribe(
                audio_path,
                language=lang_opt if lang_opt and lang_opt != "auto" else None,
                initial_prompt=prompt,
                vad_filter=True,
                beam_size=5,
            )
            first_seg = None
            it = iter(segs_gen)
            try:
                first_seg = next(it)
            except StopIteration:
                pass
        else:
            raise e
    print(f"[+] Detected language: {info.language} ({info.language_probability:.1%}), Duration: {info.duration:.1f}s")
    all_segs = []
    if first_seg is not None:
        all_segs.append(first_seg)
    all_segs.extend(list(it))
    segments = []
    texts = []
    srt_parts = []
    for idx, s in enumerate(all_segs, start=1):
        clean_text = s.text.strip()
        if not clean_text:
            continue
        segments.append({"id": idx, "start": s.start, "end": s.end, "text": clean_text})
        texts.append(clean_text)
        s_start = format_timestamp(s.start, srt=True)
        s_end = format_timestamp(s.end, srt=True)
        srt_parts.append(f"{idx}\n{s_start} --> {s_end}\n{clean_text}\n")
    return {
        "language": info.language,
        "duration": info.duration,
        "full_text": "\n".join(texts),
        "srt": "\n".join(srt_parts),
        "segments": segments,
    }


def sanitize_filename(name: str) -> str:
    clean = re.sub(r'[\\/*?:"<>|]', "_", name).strip().strip(".")
    return clean[:60] if clean else "transcript"


def main():
    parser = argparse.ArgumentParser(
        description="Extract complete spoken speech content from video URLs or local files to text."
    )
    parser.add_argument("url", nargs="?", default=None, help="Video URL or local media file path")
    parser.add_argument("-o", "--output", default=None, help="Output destination path")
    parser.add_argument("-f", "--format", choices=["txt", "srt", "json", "all"], default="txt", help="Output format")
    parser.add_argument(
        "-m", "--model",
        choices=["tiny", "base", "small", "medium", "large-v3-turbo"],
        default="small",
        help="Whisper model size",
    )
    parser.add_argument("-l", "--lang", choices=["auto", "zh", "en"], default="auto", help="Spoken language")
    parser.add_argument("-t", "--timestamps", action="store_true", help="Print timestamps alongside speech")
    parser.add_argument("-d", "--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device")
    parser.add_argument("-c", "--copy", action="store_true", help="Copy transcribed content to clipboard")
    args = parser.parse_args()
    input_target = args.url
    if not input_target:
        try:
            input_target = input("Enter video URL or local media file path: ").strip()
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)
    if not input_target:
        print("Error: No input link or file provided.")
        sys.exit(1)
    temp_dir = tempfile.mkdtemp(prefix="extract_speech_")
    try:
        print("[1/3] Fetching media stream...")
        audio_file, video_title = download_audio(input_target, temp_dir)
        print(f"[+] Audio ready: {video_title}")
        print("\n[2/3] Transcribing spoken words...")
        res = transcribe_audio(
            audio_path=audio_file,
            model_size=args.model,
            device_opt=args.device,
            lang_opt=args.lang,
        )
        full_text = res["full_text"]
        srt_content = res["srt"]
        segments = res["segments"]
        print("\n" + "=" * 60)
        print("Transcribed Spoken Content:")
        print("=" * 60)
        if not segments:
            print("(No spoken voice detected)")
        else:
            if args.timestamps:
                for seg in segments:
                    t1 = format_timestamp(seg["start"])
                    t2 = format_timestamp(seg["end"])
                    print(f"[{t1} -> {t2}] {seg['text']}")
            else:
                print(full_text)
        print("=" * 60 + "\n")
        if args.copy and HAS_CLIPBOARD:
            try:
                pyperclip.copy(full_text)
                print("[+] Full text copied to clipboard.")
            except Exception:
                pass
        print("[3/3] Saving transcription...")
        base_name = sanitize_filename(video_title)
        if args.output:
            out_base, ext = os.path.splitext(args.output)
            fmts = [ext.replace(".", "").lower()] if ext else [args.format]
            if args.format == "all":
                fmts = ["txt", "srt", "json"]
        else:
            out_base = os.path.abspath(base_name)
            fmts = ["txt", "srt", "json"] if args.format == "all" else [args.format]
        for fmt in fmts:
            target_path = f"{out_base}.{fmt}"
            if fmt == "txt":
                with open(target_path, "w", encoding="utf-8") as f:
                    if args.timestamps:
                        for seg in segments:
                            t1 = format_timestamp(seg["start"])
                            t2 = format_timestamp(seg["end"])
                            f.write(f"[{t1} -> {t2}] {seg['text']}\n")
                    else:
                        f.write(full_text)
                print(f"[+] Text saved: {target_path}")
            elif fmt == "srt":
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                print(f"[+] Subtitles saved: {target_path}")
            elif fmt == "json":
                with open(target_path, "w", encoding="utf-8") as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)
                print(f"[+] JSON saved: {target_path}")
        print("\nDone.")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
