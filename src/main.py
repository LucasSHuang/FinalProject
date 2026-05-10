import os
import pickle
import tkinter as tk
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image, ImageTk
import webbrowser
import urllib.request
import urllib.parse
import json

# ---------- Back End ----------

# preset weights for the neural network, basically all of the training has been done in this library to recognize faces
app = FaceAnalysis(name = "buffalo_l")

# ctx_id = -1 makes it so that you use the CPU and det_size is the input resolution
app.prepare(ctx_id = -1, det_size = (320, 320))

class Player:

    # init a player with their name, cropped image, and vector embedding
    def __init__(self, name, image_path = None, image = None):
        self.name = name
        self.full_image = None
        self.embedding = None
        # Accepts live camera frame from user or image path for the player images to load on disk
        if image is not None:
            full_image = image
        elif image_path is not None:
            full_image = cv2.imread(image_path)
        else:
            return

        # If there isn't an image then return nothing
        if full_image is None:
            return

        # Get all the data from the image and if there are no faces then also return nothing
        faces = app.get(full_image)
        if len(faces) == 0:
            return

        # Get the largest face in the image and turn it into a vector embedding
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        self.embedding = face.embedding
        self.full_image = full_image

    # Compares image to another image and returns how similar they are
    def compare(self, other_player):
        a = self.embedding
        b = other_player.embedding
        if a is None or b is None:
            return 0.0

        # Equation that returns value between -1 and 1, 1 being identical
        similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return similarity

class Agent:

    # Initialize agent with a hashmap to hold players
    # Threshold of 0.45 for cosign similarity because if it is that similar it is usually a match
    def __init__(self, threshold = 0.45):
        self.threshold = threshold
        self.roster = {}

    # Load directory for the first time
    def load_directory(self, manifest_path):

        # Open up text file and parse it by ", " and then register the player
        with open(manifest_path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                name, path = line.split(", ")
                self.register(name, path)

    # Add player into hashmap
    def register (self, name, image_path):

        # Initialize player and add it to hashmap
        try:
            player = Player(name, image_path)
            if player.embedding is not None:
                self.roster[name] = player
            else:
                print(f"Invalid Image for {name}")
        except FileNotFoundError as e:
            print(e)

    # Saves all of the players onto a disk by compressing embedding and image into binary
    def save_roster(self, cache_path):
        # Iterates through every player in hashmap then dumps them into the cache
        data = {name: (p.embedding, p.full_image) for name, p in self.roster.items()}
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    # Basically uncompresses the cache so this way you aren't initializing every player with the image path every time
    def load_roster(self, cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        for name, (embedding, full_image) in data.items():
            player = Player.__new__(Player)
            player.name = name
            player.embedding = embedding
            player.full_image = full_image
            self.roster[name] = player

    # Returns the most identical player to frame if there is one
    def identify(self, frame):
        unknown = Player("unknown", image = frame)

        # If there isn't a face there isn't a match
        if unknown.embedding is None:
            return None
        best_player = None
        best_score = 0.0

        # Go through every player in the hashmap and compare the image frame to the player
        for player in self.roster.values():
            score = unknown.compare(player)
            if score > best_score:
                best_score = score
                best_player = player
        if best_score >= self.threshold:
            return best_player
        return None

# ---------- Front End ---------

BG = "#0f1115"
PANEL = "#171a21"
BLUE = "#3b82f6"
BLUE_HOVER = "#2563eb"
RED = "#ef4444"
RED_HOVER = "#dc2626"
TEXT = "#e5e7eb"
MUTED = "#9ca3af"
GREEN = "#4ade80"
YELLOW = "#fbbf24"
YELLOW_HOVER = "#f59e0b"

DISPLAY_W = 640
DISPLAY_H = 480


class App:
    def __init__(self, root, agent):

        # Root window and agent that holds the roster
        self.root = root
        self.agent = agent

        # State of if there is a matched player yet and frame counter so that I don't run identification every frame
        self.matched_player = None
        self.frame_count = 0

        # Window setup
        root.title("NBA Player Identifier")
        root.configure(bg = BG)
        root.geometry("760x760")
        root.protocol("WM_DELETE_WINDOW", self.quit)

        # Header title at top of window
        tk.Label(
            root, text = "NBA Player Identifier",
            font=("Helvetica", 22, "bold"),
            bg = BG, fg = TEXT,
        ).pack(pady = (24, 4))

        # Status text under header so initially searching but StringVar allows for changing the text later
        self.status_var = tk.StringVar(value = "Searching for player…")
        self.status_label = tk.Label(
            root, textvariable = self.status_var,
            font = ("Helvetica", 13),
            bg = BG, fg = GREEN,
        )
        self.status_label.pack(pady = (0, 16))

        # Container frame for the camera display
        self.panel = tk.Frame(root, bg = PANEL, width = DISPLAY_W + 24, height = DISPLAY_H + 24)
        self.panel.pack(padx = 24)
        self.panel.pack_propagate(False)

        # Actual canvas for where camera or matched player is put
        self.canvas = tk.Canvas(
            self.panel, width = DISPLAY_W, height = DISPLAY_H,
            bg = PANEL, highlightthickness = 0, bd = 0,
        )
        self.canvas.pack(expand=True)

        # ID of image on canvas so that we don't have to keep stacking images and instead change the image
        self.canvas_image_id = None

        # Matched player name but empty until match is found
        self.name_var = tk.StringVar(value = "")
        self.name_label = tk.Label(
            root, textvariable = self.name_var,
            font = ("Helvetica", 26, "bold"),
            bg = BG, fg = TEXT,
        )
        self.name_label.pack(pady = (20, 8))

        # Buttons
        button_frame = tk.Frame(root, bg = BG)
        button_frame.pack(pady = (8, 24))

        # Rescan clears current match and goes back to camera
        self.rescan_btn = self._make_button(button_frame, "Rescan", BLUE, BLUE_HOVER, self.reset)
        self.rescan_btn.pack(side="left", padx=8)

        # Quit stops the camera and closes the app
        self.quit_btn = self._make_button(button_frame, "Quit", RED, RED_HOVER, self.quit)
        self.quit_btn.pack(side="left", padx=8)

        # ESPN button created here but only shows up if there is a match
        self.espn_btn = self._make_button(button_frame, "View on ESPN", YELLOW, YELLOW_HOVER, self.open_espn)

        # Opens up the camera and if fails displays an error message
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.status_var.set("Could not open camera")
            self.status_label.configure(fg = RED)
            return

        # Recursive frame loop
        self.update_frame()

    # Makes the buttons
    def _make_button(self, parent, text, color, hover, cmd):
        btn = tk.Label(
            parent, text = text,
            font = ("Helvetica", 12, "bold"),
            bg = color, fg = "white",
            padx = 24, pady = 10, cursor = "hand2",
        )

        # Creates a hover effect where the color of the button changes when you enter it
        btn.bind("<Enter>", lambda e: btn.configure(bg = hover))
        btn.bind("<Leave>", lambda e: btn.configure(bg = color))

        # When you click it the command activates
        btn.bind("<Button-1>", lambda e: cmd())
        return btn

    # Resets everything back to inital state
    def reset(self):
        self.matched_player = None
        self.name_var.set("")
        self.status_var.set("Searching for player…")
        self.status_label.configure(fg = GREEN)
        self.espn_btn.pack_forget()

    # Release the camera and then destroy the window
    def quit(self):
        if hasattr(self, "camera") and self.camera.isOpened():
            self.camera.release()
        self.root.destroy()

    # Repeatedly updates the frame
    def update_frame(self):

        if self.matched_player is None:
            # If there isn't a matched player grab a new frame
            ret, frame = self.camera.read()
            if ret:
                self.frame_count += 1

                # Run the identification part every 50 frames
                if self.frame_count % 50 == 0:
                    found = self.agent.identify(frame)

                    # If there is a match update the UI and add ESPN button
                    if found is not None:
                        self.matched_player = found
                        self.status_var.set("Match found")
                        self.status_label.configure(fg = YELLOW)
                        self.name_var.set(found.name)
                        self.espn_btn.pack(side = "left", padx = 8, before = self.quit_btn)
                self._show_image(frame)

        # Now just show the photo of the matched player instead of the camera
        else:
            self._show_image(self.matched_player.full_image)

        # Iterate every 15ms
        self.root.after(15, self.update_frame)

    # Shows the camera frame or player photo
    def _show_image(self, img):
        if img is None or img.size == 0:
            return

        # Blurs some of the blue dots so they aren't as annoying in player headshot
        img = cv2.bilateralFilter(img, d = 5, sigmaColor = 40, sigmaSpace = 5)

        # Get largest ratio for image to fit inside display area
        h, w = img.shape[:2]
        scale = min(DISPLAY_W / w, DISPLAY_H / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Letterboxing to make sure the image always appears in the middle instead of jumping around
        letterbox = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
        letterbox[:] = (33, 26, 23)  # matches PANEL roughly in BGR
        y_off = (DISPLAY_H - new_h) // 2
        x_off = (DISPLAY_W - new_w) // 2
        letterbox[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # Makes sure colors are read in right order because openCV reads in BGR instead of RGB
        rgb = cv2.cvtColor(letterbox, cv2.COLOR_BGR2RGB)

        # Converts numpy array to image path
        pil_img = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(pil_img)

        # On first frame create canvas image item
        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(
                DISPLAY_W // 2, DISPLAY_H // 2, image = tk_img
            )
        # Afterward just swap the image of the existing item
        else:
            self.canvas.itemconfig(self.canvas_image_id, image = tk_img)

        # To ensure canvas doesn't go blank
        self.canvas.image = tk_img

    # Opens matched player's ESPN page
    def open_espn(self):
        if self.matched_player is None:
            return
        try:
            # URL encode name of player
            query = urllib.parse.quote(self.matched_player.name)
            url = f"https://site.web.api.espn.com/apis/search/v2?query={query}&limit=5&type=player"

            # ESPN blocks normal python user agent so disguise it as Chrome
            req = urllib.request.Request(url, headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
            })

            # API call to ESPN's hidden search API and then parses the response
            with urllib.request.urlopen(req, timeout = 5) as resp:
                data = json.loads(resp.read())

            # Find the first NBA player hit and use its real web link
            for group in data.get("results", []):
                if group.get("type") != "player":
                    continue
                for hit in group.get("contents", []):
                    if hit.get("defaultLeagueSlug") == "nba":
                        web_url = hit.get("link", {}).get("web")
                        if web_url:
                            webbrowser.open(web_url)
                            return

            # If mo NBA player found just search up player name on ESPN
            webbrowser.open(f"https://www.espn.com/search/_/q/{query}")

        # If there is an error somehow state it
        except Exception as e:
            print(f"ESPN lookup failed: {e}")

# ---------- Main Code ----------

# Build agent that holds all the player embeddings and images
agent = Agent()

# Path to player text file
manifest = "/Users/mba/IdeaProjects/FinalProject/resources/players.txt"

# Path to cache with pickled forms of embeddings and images which is faster than initializing every time
cache = "/Users/mba/IdeaProjects/FinalProject/resources/roster_cache.pkl"

if os.path.exists(cache):
    # If cache is there just load embeddings and images from disk
    agent.load_roster(cache)
else:
    # Otherwise just intialize all the players and save them into the cache for next time
    agent.load_directory(manifest)
    agent.save_roster(cache)

# Create the root window
root = tk.Tk()

# Launch the app
App(root, agent)

# Keep going until quit is pressed or manually stopped
root.mainloop()