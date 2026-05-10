This project is a real time face recognition app that uses a pre-trained AI model to recognize NBA players from a webcam

The app loads a roster of every current NBA player and converts each player's headshot into a 512-dimensional vector using InsightFace's `buffalo_l` face recognition model.

When you start the app, it streams video from your webcam through a Tkinter GUI.
Every 50 frames, it grabs a frame, runs the same face recognition on it, and compares the result to every player in the roster using cosine similarity.
If the closest match scores above a 0.45 threshold, which means the faces are "similar enough" the app:

1. Switches the camera display to that player's official headshot
2. Shows their name underneath
3. Reveals a "View on ESPN" button that opens their actual ESPN player page

The ESPN integration uses ESPN's undocumented public search API.
Instead of pre-loading IDs for every player (impractical for 340+ players),
the app sends the matched player's name to ESPN's search endpoint at runtime,
parses the response, and pulls out the direct link to their profile page.