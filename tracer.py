# tracer.py

from agents import add_trace_processor
from agents.tracing.processor_interface import TracingProcessor


class GradioTraceProcessor(TracingProcessor):

    def __init__(self):
        self.events = []

    def clear(self):
        self.events.clear()

    def get_text(self):
        if not self.events:
            return "No trace events recorded."

        return "\n".join(self.events)

    # =====================================================
    # Trace Events
    # =====================================================

    def on_trace_start(self, trace):
        self.events.append(
            f"🚀 TRACE START | {trace.name}"
        )

    def on_trace_end(self, trace):
        self.events.append(
            f"✅ TRACE END | {trace.trace_id}"
        )

    # =====================================================
    # Span Events
    # =====================================================

    def on_span_start(self, span):

        span_type = type(span.span_data).__name__

        self.events.append(
            f"▶️ START | {span_type}"
        )

    def on_span_end(self, span):

        span_type = type(span.span_data).__name__

        self.events.append(
            f"✔️ END | {span_type}"
        )

    # =====================================================
    # Required Methods
    # =====================================================

    def shutdown(self):
        pass

    def force_flush(self):
        pass


trace_processor = GradioTraceProcessor()

add_trace_processor(trace_processor)