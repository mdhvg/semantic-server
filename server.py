import socket
import threading
import json
from sentence_transformers import SentenceTransformer
from collections import deque
from typing import List, Dict, Tuple, Literal
import torch
import time
import struct


class Response:
    def __init__(self, id: int, vector: torch.Tensor, is_query: bool):
        self.id = id
        self.is_query = is_query
        self.vector = vector


class Message:
    def __init__(self, kind: Literal["DATA", "COMMAND", "QUERY"], **kwargs):
        self.kind = kind
        if kind == "DATA":
            self.id = kwargs["id"]
            self.data = kwargs["data"]
        elif kind == "COMMAND":
            self.command = kwargs["command"]
        elif kind == "QUERY":
            self.data = kwargs["data"]
        else:
            raise ValueError("Invalid message type", kind, kwargs)


running: threading.Event = threading.Event()
running.set()
# model = [SentenceTransformer("all-MiniLM-L6-v2")]
modelLock: threading.Lock = threading.Lock()
model = []
embeddingQ: deque["Response"] = deque()
embeddingQLock: threading.Lock = threading.Lock()


def model_loader(modelArr: List, modelLock: threading.Lock) -> None:
    start = time.time()
    modelLock.acquire()
    modelArr.append(SentenceTransformer("all-mpnet-base-v2"))
    modelLock.release()
    print(f"Model loading took: {time.time() - start}s")


def run_command(command: str) -> None:
    if command == "close":
        running.clear()
        print("Server: Quitting")
        return
    if command == "heartbeat":
        # print(f"Server: {time.time():.4f} Heartbeat")
        return


def process_message(message: Message) -> None:
    if message.kind == "COMMAND":
        run_command(message.command)
        return
    modelLock.acquire()
    embedding = model[0].encode([message.data])
    modelLock.release()
    if message.kind == "QUERY":
        embeddingQ.append(Response(0, embedding, True))
        return
    if message.kind == "DATA":
        embeddingQ.append(Response(message.id, embedding, False))
        return


def embedding_response(
    embeddingQ: deque["Response"],
    running: threading.Event,
    client_socket: socket.socket,
    queuelock: threading.Lock,
) -> None:
    while running.is_set():
        if len(embeddingQ) > 0:
            queuelock.acquire()
            embedding = embeddingQ.popleft()
            queuelock.release()
            response = json.dumps(
                {
                    "id": embedding.id,
                    "isQuery": embedding.is_query,
                    "vector": embedding.vector.tolist()[0],
                }
            )
            length = len(response)
            client_socket.sendall(struct.pack(">I", length))
            client_socket.sendall(response.encode("utf-8"))
        else:
            time.sleep(1)
    print("Closing response thread")


def handle_client_connection(client_socket: socket.socket) -> None:
    client_socket.settimeout(5)
    # NOTE: Use a heartbeat mechanism from client to keep the connection alive, or the current socket will die after 5 seconds of inactivity
    while running.is_set():
        try:
            header = client_socket.recv(4)
            if not header:
                continue
            length = struct.unpack(">I", header)[0]

            request = client_socket.recv(length).decode("utf-8")
            print(request)
            request_content = json.loads(request)
            try:
                message: Message = Message(**request_content)
                process_message(message)
            except Exception as e:
                print(f"ValueError: {e}")
                running.clear()
        except Exception as e:
            running.clear()
            print(f"Error: {e}")
            break

    print(f"Closing client connection {client_socket.getpeername()}")
    client_socket.close()


def start_server(limit: int = 2):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", 5096))
    server.settimeout(5)
    server.listen(5)
    clients: List[socket.socket] = []
    client_threads: List[Tuple[threading.Thread, threading.Thread]] = []
    runs = 0
    print("Python server listening on port 5096")

    while running.is_set():
        try:
            client_socket, addr = server.accept()
            print(f"Accepted connection from {addr}")
            clients.append(client_socket)
            client_handler: threading.Thread = threading.Thread(
                target=handle_client_connection, args=(client_socket,)
            )
            response_thread: threading.Thread = threading.Thread(
                target=embedding_response,
                args=(embeddingQ, running, client_socket, embeddingQLock),
            )
            client_threads.append((client_handler, response_thread))
            client_handler.start()
            response_thread.start()
        except TimeoutError:
            if runs >= limit and len(clients) == 0:
                print("Server running idle, Shutting down.")
                running.clear()
            runs += 1
            continue
        print(clients)

    for client_thread, response_thread in client_threads:
        client_thread.join()
        response_thread.join()
    print("Closing server")
    server.close()


if __name__ == "__main__":
    try:
        modelThread: threading.Thread = threading.Thread(
            target=model_loader, args=(model, modelLock)
        )
        modelThread.start()
        start_server()
        modelThread.join()
        model.clear()
    except Exception as e:
        running.clear()
        print(f"Error: {e}")
        exit(1)
