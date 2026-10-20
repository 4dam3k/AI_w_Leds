import asyncio
from pprint import pprint

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

    """Show example of controlling your WLED device."""
    async with WLED(ip_address) as led:
        device = await led.update()
        dev_effects = device.effects
        effects = [{
            "name": dev_effects[i].name,
            "palette": dev_effects[i].metadata.palette,
        } for i in dev_effects.keys()]
        # for x in dev_effects.keys():
        #     pprint(dev_effects[x])
        dev_palettes = device.palettes
        # for x in dev_palettes.keys():
        #     pprint(dev_palettes[x])
        # while True:
        effect_chosen = await asyncio.gather(lm(effects, dev_palettes))
        effect_chosen = effect_chosen[0]

        if not effect_chosen["can_have_palette"]:
            effect_chosen["palette"] = False
        effect_chosen["primary_color"] = ImageColor.getcolor(effect_chosen["primary_color"], "RGB")

        pprint(effect_chosen)
        print(dev_palettes[effect_chosen["palette"]])
        print(effect_chosen["brightness"])
        print(type(effect_chosen["brightness"]))
        await led.segment(
            0,
            effect= effect_chosen["name"],
            palette= effect_chosen["palette"],
            color_primary= effect_chosen["primary_color"],
            brightness= int(effect_chosen["brightness"]),
        )
        await led.master(on=True)


class EffectShema(BaseModel):
    name: str
    can_have_palette: bool
    brightness: int
    palette: int
    primary_color: str


async def lm(effects: list, palettes: dict):
    async with lms.AsyncClient() as client:
        model = await client.llm.model("qwen/qwen3.5-9b")

        session = PromptSession()
        with patch_stdout():
            effect_prompt = await session.prompt_async("Tell me what effect do you need?\n    ")

        # 1. SYSTEM PROMPT: Defines role, rules, and provides context data
        system_prompt = f"""You are a WLED lighting control assistant.
        Your task is to select the best lighting effect and its parameters based on the user's request.

Available effects:
{json.dumps(effects, indent=2)}

Available palettes (key -> name):
{json.dumps(palettes, indent=2)}

Rules:
1. Choose an effect name EXACTLY from the available effects list.
2. If the effect supports a palette, select the best matching palette ID (the dictionary key) from the available palettes.
3. If the effect does NOT support a palette, set "palette" to false and provide a primary color as a HEX string (e.g., "#FF5733").
4. Brightness must be an integer between 0 and 255.
5. Return ONLY a valid JSON object matching this structure:
{{
  "name": "<exact effect name>",
  "can_have_palette": true/false,
  "brightness": <0-255>,
  "palette": "<palette_id>" or false,
  "primary_color": "<HEX string if no palette>"
}}"""

        # 2. USER PROMPT: Contains the dynamic user request
        user_prompt = f"Effect I need is: {effect_prompt}"

        # 3. MESSAGES ARRAY: Standard OpenAI/LM Studio message format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Pass messages instead of a raw string.
        # response_format expects the Pydantic JSON schema
        prediction = await model.respond_stream(
            messages=messages,
            response_format=EffectShema.model_json_schema()
        )

        async for fragment in prediction:
            print(fragment.content, end="", flush=True)
        print()

        # LM Studio may return a dict or a JSON string. Handle both safely.
        raw_result = prediction.result().content if hasattr(prediction, 'result') else str(prediction)
        effect_chosen = raw_result if isinstance(raw_result, dict) else json.loads(raw_result)

        print("Parsed result:\n", effect_chosen, "\n")
        return effect_chosen



if __name__ == "__main__":
    asyncio.run(main())

