import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# >>> add this import
from interrupt_filter import InterruptFilter

logger = logging.getLogger("basic-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "do not use emojis, asterisks, markdown, or other special characters in your responses."
            "You are curious and friendly, and have a sense of humor."
            "you will speak english to the user",
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
    )

    # >>> create filter + a flag to track if agent is talking
    filler_filter = InterruptFilter()
    agent_speaking = False

    # >>> mark when the agent starts/stops speaking by watching conversation items
    # this fires for both user and agent text; we use it to guess speaking state
    @session.on("conversation_item_added")
    def _on_conv_item(ev):
        nonlocal agent_speaking
        # if the item was produced by the agent itself, we're speaking
        if ev.item.sender == "agent":
            agent_speaking = True
            # when the agent finishes that item, LiveKit sends final=True
            if getattr(ev.item, "final", False):
                agent_speaking = False
        else:
            # user item → user is talking
            pass

    # >>> this is the actual place to filter interruptions
    @session.on("transcription")
    async def _on_user_transcription(ev):
        nonlocal agent_speaking
        transcript = ev.text
        confidence = getattr(ev, "confidence", 1.0)

        # ask our module if this should interrupt
        if await filler_filter.is_meaningful(transcript, confidence, agent_speaking):
            # stop current TTS
            session.interrupt()
            print(f"[INTERRUPT] {transcript}")
        else:
            print(f"[IGNORED] {transcript}")

    # metrics (your original code)
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
