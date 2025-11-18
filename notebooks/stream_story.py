from pydantic_ai import Agent
from pathlib import Path
from dotenv import load_dotenv
import asyncio

load_dotenv(Path(__file__).parents[1].joinpath('.env'))

# Storyteller Agent
story_agent = Agent(
    'openai:gpt-5-mini',
    system_prompt="You are an AI storyteller. Generate engaging, real-time sci-fi adventures."
)

# Stream the story
async def stream_story():
    user_prompt = "Tell me a sci-fi story about a lost spaceship in a short response."
    async with story_agent.run_stream(user_prompt) as response:
        async for part in response.stream_text():
            print(part, end='', flush=True)

# Run the streaming story generator
asyncio.run(stream_story())