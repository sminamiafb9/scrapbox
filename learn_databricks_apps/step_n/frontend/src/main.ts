import { PipecatClient } from "@pipecat-ai/client-js";
import {
  WebSocketTransport,
  ProtobufFrameSerializer,
} from "@pipecat-ai/websocket-transport";

const transport = new WebSocketTransport({
  serializer: new ProtobufFrameSerializer(),
  recorderSampleRate: 16000,
  playerSampleRate: 16000,
});

const client = new PipecatClient({
  transport,
  enableCam: false,
  enableMic: true,
});

const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";

const wsUrl = `${wsProtocol}//${location.host}/ws`;

export async function start() {
  console.log("Connecting:", wsUrl);

  await client.connect({
    wsUrl,
  });

  console.log("Connected");
}

export async function stop() {
  console.log("Disconnecting");

  await client.disconnect();

  console.log("Disconnected");
}

(window as any).startConversation = start;
(window as any).stopConversation = stop;

start().catch(console.error);