# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Own live modal operators only until cancellation or confirmation."""

from types import SimpleNamespace

import bpy


_sessions = []


class ModalSession:
    def __init__(self, operator, context):
        self.operator = operator
        # Context itself is dynamic. Retain the original UI owners for cleanup
        # even when unregister is called from another area or the Python console.
        self.context = SimpleNamespace(
            area=context.area, window=context.window,
            workspace=context.workspace, window_manager=context.window_manager,
        )
        self.closed = False

    def close(self, cancel=False):
        if self.closed:
            return
        self.closed = True
        _sessions.remove(self)
        operator, context = self.operator, self.context
        try:
            if cancel:
                if isinstance(context.window, bpy.types.Window):
                    with bpy.context.temp_override(window=context.window):
                        operator.cancel(context)
                else:
                    operator.cancel(context)
        finally:
            try:
                operator._modal_session = None
            finally:
                self.operator = self.context = None


def begin_modal_session(operator, context):
    session = ModalSession(operator, context)
    operator._modal_session = session
    _sessions.append(session)


def end_modal_session(operator):
    session = getattr(operator, '_modal_session', None)
    if session is not None:
        session.close()


def cancel_modal_sessions():
    """Cancel before unregistering RNA classes, then release every strong owner."""
    for session in tuple(_sessions):
        try:
            session.close(cancel=True)
        except (ReferenceError, RuntimeError, ValueError) as error:
            # Continue releasing other sessions/UI resources if Blender has
            # already removed an object, workspace, or operator during reload.
            print(f'Holey Tools: modal cleanup failed: {error}')
