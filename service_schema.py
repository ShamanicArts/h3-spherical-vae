"""Public request contract; validation requires no GPU or Comfy imports."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPAIR_PROMPT = 'Repair the polar distortion.'


class SphericalRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(default='', max_length=8000, description='Main panorama generation prompt. Never copied into pole repair instructions.')
    video_url: str | None = Field(default=None, description='Optional existing complete ERP video to repair. Exactly 124 frames at 24 fps; no resizing or trimming.')
    width: Literal[1024, 1536] = 1536
    height: Literal[448, 672] = 672
    frames: Literal[124] = 124
    steps: int = Field(default=50, ge=1, le=100)
    seed: int = Field(default=2026090919, ge=0, le=2**63-1)
    circular_decode: bool = True
    context_pixels: Literal[128, 384] = 128
    final_shift: bool = True
    repair_top: bool = True
    repair_bottom: bool = True
    top_prompt: str = Field(default=REPAIR_PROMPT, max_length=8000)
    bottom_prompt: str = Field(default=REPAIR_PROMPT, max_length=8000)
    repair_steps: int = Field(default=32, ge=1, le=40)
    repair_seed: int = Field(default=2026090911, ge=0, le=2**63-1)

    @model_validator(mode='after')
    def coherent_input(self):
        if (self.width, self.height) not in ((1024,448),(1536,672)):
            raise ValueError('Choose 1024x448 or 1536x672; dimensions are never silently adjusted')
        if self.video_url:
            if not self.video_url.startswith('https://'):
                raise ValueError('video_url must be HTTPS')
            if self.prompt.strip():
                raise ValueError('Existing-video repair does not use a generation prompt; leave prompt empty')
        elif not self.prompt.strip():
            raise ValueError('Provide a generation prompt or video_url')
        return self

    def generation_config(self):
        return dict(prompt=self.prompt, width=self.width, height=self.height,
                    frames=self.frames, steps=self.steps, seed=self.seed, fps=24,
                    contexts=[self.context_pixels if self.circular_decode else 0],
                    final_shift=self.final_shift, lora_strength=1.0)

    def poles(self):
        return [(name,pitch,prompt) for name,pitch,prompt,enabled in
                [('top',90,self.top_prompt,self.repair_top),
                 ('bottom',-90,self.bottom_prompt,self.repair_bottom)] if enabled]
