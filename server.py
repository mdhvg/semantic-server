import socket
import threading
import json
from sentence_transformers import SentenceTransformer
from collections import deque
from typing import List, Dict, Tuple
import torch
import time


class Document:
    def __init__(self, size: int, data: str = ""):
        self.size = size
        self.data = data
        self.pending = size


class Response:
    def __init__(self, id: str, vector: torch.Tensor, is_query: bool):
        self.id = id
        self.is_query = is_query
        self.vector = vector


running: threading.Event = threading.Event()
running.set()
# model = [SentenceTransformer("all-MiniLM-L6-v2")]
modelLock: threading.Lock = threading.Lock()
model = []
embeddingQ: deque["Response"] = deque()
embeddingQLock: threading.Lock = threading.Lock()
delim = ";;"
ending = "$$"
idMap: Dict[str, "Document"] = {}


def model_loader(modelArr: List, modelLock: threading.Lock) -> None:
    start = time.time()
    modelLock.acquire()
    modelArr.append(SentenceTransformer("all-mpnet-base-v2"))
    modelLock.release()
    print(f"Model loading took: {time.time() - start}s")


def parse_command(command):
    if command == "Quit":
        running.clear()
        print("Server: Quitting")
    if command == "Heartbeat":
        print(f"Server: {time.time():.4f} Heartbeat")


def process_data(id: str, data: str, is_query: bool = False) -> None:
    print(f"Processing data with id {id}")
    if id not in idMap:
        print(f"Adding data with id {id}")
        idMap[id] = Document(int(data))
    else:
        idMap[id].data += data
        idMap[id].pending -= len(data)
        print(idMap[id].pending)
        print(idMap[id].size)
        print(idMap[id].data)
        if idMap[id].pending == 0:
            modelLock.acquire()
            embedding = model[0].encode([idMap[id].data])
            modelLock.release()
            embeddingQ.append(Response(id, embedding, is_query))
            del idMap[id]


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
            client_socket.send(response.encode("utf-8"))
        else:
            time.sleep(1)
    print("Closing response thread")


def handle_client_connection(client_socket: socket.socket) -> None:
    client_socket.settimeout(5)
    # NOTE: Use a heartbeat mechanism from client to keep the connection alive, or the current socket will die after 5 seconds of inactivity
    while running.is_set():
        try:
            request = client_socket.recv(4096).decode("utf-8")
            if not len(request):
                continue
            request_content = request.split(ending)[0]
            print(request_content)
            try:
                reqType, id, payload = request_content.split(delim)
            except Exception as e:
                print(f"ValueError: {e}")
                running.clear()
            if reqType == "Q":
                process_data(id, payload, True)
                continue
            if reqType == "C":
                parse_command(payload)
                continue
            if reqType == "D":
                process_data(id, payload)
                continue
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
