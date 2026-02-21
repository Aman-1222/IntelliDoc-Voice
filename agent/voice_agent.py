import os
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)

from livekit.plugins import deepgram, elevenlabs, groq

from agent.agent_core import run_agent

load_dotenv()


class IntelliDocAgent(Agent):

    def __init__(self):
        super().__init__(
            instructions="You are IntelliDoc Voice."
        )

    async def on_conversation_item_added(self, event):
        if not hasattr(event, "item"):
            return

        if event.item.role != "user":
            return

        transcript = event.item.text_content
        if not transcript:
            return

        print(f"\nUser said: {transcript}")

        # 🔥 Call your existing Groq multi-step agent
        response = run_agent(transcript)

        print(f"Agent response: {response}")

        await self.session.say(response)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(),
        llm=groq.LLM(model="llama-3.1-8b-instant"),
        tts=elevenlabs.TTS(),
    )

    agent = IntelliDocAgent()

    await session.start(room=ctx.room, agent=agent)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )