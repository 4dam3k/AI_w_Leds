import asyncio
from pprint import pprint
from typing import Optional

from wled import WLED
from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf
import socket
import lmstudio as lms
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from pydantic import BaseModel
import json
from PIL import ImageColor


async def resolve_mdns(device_name: str) -> str:
    aio_zc = AsyncZeroconf()
    try:
        name = device_name.split('.')[0]
        type_ = "_http._tcp.local."
        service_name = f"{name}.{type_}"

        info = AsyncServiceInfo(type_, service_name)
        await info.async_request(aio_zc.zeroconf, 3000)

        if info.addresses:
            return socket.inet_ntoa(info.addresses[0])

        raise EnvironmentError(f"Unable to resolve device: {device_name}")
    finally:
        await aio_zc.async_close()


async def main() -> None:
    ip_address = await resolve_mdns("buubuu.local")

    async with WLED(ip_address) as led:
        device = await led.update()

        # Simplify effects list
        effects = [
            {
                "name": effect.name,
                "palette": effect.metadata.palette,
            }
            for effect in device.effects.values()
        ]

        # Map palette_id -> name (ignoring custom field)
        dev_palettes = {
            palette.palette_id: palette.name
            for palette in device.palettes.values()
        }

        effect_chosen = await lm(effects, dev_palettes)
        # pprint(effect_chosen)

        # Parse primary color to RGB tuple
        rgb_color = ImageColor.getcolor(effect_chosen["primary_color"], "RGB")

        segment_kwargs = {
            "effect": effect_chosen["name"],
            "color_primary": rgb_color,
        }

        # If the user chose a palette, apply it; otherwise, use palette 0 (Default) to ensure solid primary colors work
        if effect_chosen.get("uses_palette") and effect_chosen.get("palette_id") is not None:
            segment_kwargs["palette"] = effect_chosen["palette_id"]
            print(f"Selected palette: {dev_palettes.get(effect_chosen['palette_id'], 'Unknown')}")
        else:
            segment_kwargs["palette"] = 0  # Forces WLED to use the custom primary color slot
            print(f"Using solid primary color: {effect_chosen['primary_color']}")

        await led.segment(0, **segment_kwargs)
        await led.master(on=True, brightness=int(effect_chosen["brightness"]))


class EffectSchema(BaseModel):
    name: str
    can_have_palette: bool
    uses_palette: bool
    brightness: int
    palette_id: Optional[int] = None
    primary_color: str


async def lm(effects: list, palettes: dict) -> dict:
    async with lms.AsyncClient() as client:
        # model = await client.llm.model("prism-ml/bonsai-27b")
        model = await client.llm.model("qwen/qwen3.5-9b")

        session = PromptSession()
        with patch_stdout():
            effect_prompt = await session.prompt_async("Tell me what effect do you need?\n    ")

        system_prompt = f"""You are a WLED lighting control assistant.
        Your task is to select the best lighting effect and parameters based on the user's request.

        Available effects:
        {json.dumps(effects, indent=2)}

        Available palettes (palette_id -> name):
        {json.dumps(palettes, indent=2)}

        Rules:
1. EFFECT SELECTION:
   - Select the effect from the available effects list that best fits the motion or vibe requested.
   - Set "can_have_palette" to true if the selected effect's "palette" flag is true, else false.

2. PALETTE DECISION LOGIC:
   Palette 0 is "Default" (uses the primary color or native multi-color logic). When "can_have_palette" is true:
   - SET "uses_palette": false AND "palette_id": 0 IF:
     * The user asks for a single solid color (e.g., "blink red", "breathing cyan", "no palette").
     * The effect natively generates its own multi-color look without needing an external palette (e.g., "Rainbow").
     * The request only describes a pure motion pattern with zero color or thematic keywords (e.g., "chase", "wipe", "scan"). Do NOT pick random palettes for generic motions.
   - SET "uses_palette": true AND PICK A MATCHING "palette_id" IF:
     * The user describes a theme, mood, nature scene, or color palette (e.g., "forest leaves", "fire/flame", "sunset", "ocean waves", "party vibes") that maps naturally to an available palette name.
     * The user explicitly names a palette from the list.

3. FALLBACK WHEN CANNOT USE PALETTE:
   - If "can_have_palette" is false, always set "uses_palette": false and "palette_id": null.

4. PRIMARY COLOR & BRIGHTNESS:
   - "primary_color": Provide a matching HEX string (e.g., "#228B22" for forest, "#FF0000" for red). If a palette is used, set this to the dominant theme color or "#FFFFFF".
   - "brightness": Integer from 1 to 255 (match descriptive words like "soft", "dim", "bright", or default to 128-200).
        """

        chat = lms.Chat(system_prompt)
        chat.add_user_message(f"Effect I need is: {effect_prompt}")

        prediction = await model.respond_stream(
            chat,
            response_format=EffectSchema.model_json_schema(),
        )

        async for fragment in prediction:
            print(fragment.content, end="", flush=True)
        print()

        raw_result = prediction.result().content if hasattr(prediction, 'result') else str(prediction)

        if isinstance(raw_result, str):
            # Strip any synthetic thinking delimiters or trailing text
            start_idx = raw_result.find("{")
            end_idx = raw_result.rfind("}")
            if start_idx != -1 and end_idx != -1:
                raw_result = raw_result[start_idx: end_idx + 1]

        effect_chosen = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
        return effect_chosen


if __name__ == "__main__":
    asyncio.run(main())

