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


effects = []


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
        # model = await client.llm.model("mistralai/mistral-nemo-instruct-2407")

        session = PromptSession()
        with patch_stdout():
            effect_prompt = await session.prompt_async("Tell me what effect do you need?\n    ")

        prompt = f"You are given this prompt: \"Effect I need is: {effect_prompt}\". You must choose an effect with name from the first list. If the effect can have palette, also specify pallete chosen from second list which is best for given prompt using its id, if not, specify sinle color as HEX value of it. All palettes are available for all effects which can have palette. Remember brightness can be 0-255\n"
        prompt += "\n"
        prompt += str(effects)
        prompt += "\n"
        prompt += str(palettes)
        print(prompt)

        # prompt = f""

        prediction = await model.respond_stream(
            prompt,
            response_format=EffectShema
        )

        async for fragment in prediction:
            print(fragment.content, end="", flush=True)
        print()

        effect_chosen = json.loads(prediction.result().content)
        print("Parsed result:\n", effect_chosen, "\n")
        return effect_chosen

if __name__ == "__main__":
    asyncio.run(main())

