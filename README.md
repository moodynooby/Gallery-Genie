# Gallery Genie (G.G) Inc: Because Who Can Afford Professional Editor Anyway?

#### Video Demo: https://www.youtube.com/TBD

#### Description:

Hey, welcome to **Gallery Genie** - my attempt at making image editing less painful for people like me who just want
cool effects without getting a PHD in image editing . I'm Manas, an engineering student and this is what happens when I
decide to mess around with photos instead of studying. Basically, it's a Python program that takes your boring pictures
and turns them into fun stuff like pixel art or vintage postcards. You just tell it what you want in normal words, and
it tries to figure it out and it works(sometimes).

Hence here you have a intern at Gallery Genie Inc. (GG) called G so please don't be hard on him 🥲

%% I added the "intern" character because it makes the whole thing feel less like a boring app and more like chatting
with a helpful (but slightly clueless) friend. It's my way of poking fun at how I'm still learning all this stuff
myself. %%

## The Thing that your friend G can do

Here's what G.G can do with your images. I picked effects that are fun and not too hard to code:

- **ASCII-fy** (- ᴗ- ): Turns photos into text art made of characters. Great for sharing in chats or impressing friends
  who think you're a hacker.
- **Get Aesthetic Photo** (🌫️): Adds a soft blur to make things look dreamy and polished, like those influencer posts.
- **Pencil Sketch** (🖍️): Makes it look like a quick drawing with pencil lines. Comes in handy for presentations.
- **Get Your Old-Timey Look** (📻): Gives photos that warm, faded color from old pictures. Perfect for family photos or
  history projects.
- **Get Your Evil Form** (🌈): Flips all the colors to make everything look dramatic and inverted. Fun for memes or
  Halloween.
- **8-bit Game Look** (🟩): Makes images blocky and pixelated like old video games. Takes me right back to playing Mario
  as a kid.
- **Cartoonify** (🧩): Simplifies the photo into bold colors and lines, like a comic strip character.

## How It Picks the Right Tool

The smart part is how it guesses what you mean. You type something like "make it pixelated" and it looks at keywords to
suggest the 8-bit tool. I used some basic text matching from a library called scikit-learn - nothing too crazy, just
enough to save you from scrolling through options. If it gets confused, it shows a simple numbered list so you can pick
manually.

## What's Under the Hood

I used OpenCV for all the image stuff - loading files, applying filters, saving results. It's the same library you'd use
for serious computer vision, but here it's just for fun effects like blurring or color flipping. NumPy helps with the
math behind the scenes, and I threw in some ASCII art libraries to make the interface look cool.

The flow is straightforward:

1. You give it an image file.
2. It applies whatever filter you picked.
3. Saves a new version and opens it right away so you can see the result.
4. For the ASCII one, it makes both text output and a picture file.

## Tests I Wrote

Since this is for a class project, I had to add some tests. I used pytest to check the basics:

- Does it pick the right tool when I describe it?
- What happens with bad input or missing files?
- Does the saving and opening part work without crashing?

They're not super fancy, but they catch the obvious bugs so I don't look silly in front of the professor.

## How to Get It Running

### Step 1: Install What You Need

Open your terminal and run:

```pip install -r requirements.txt ```

  
### Step 2: The Genie File  
Put a picture in the same directory as path , Here I have included `Genie.jpg` in the same folder as the script.  
  
### Step 3: Run the Program  

``` python app.py```

  
### Step 4: Using It  
- It'll greet you and ask your name (optional, but fun).  
- Type what you want, like "turn it into a cartoon."  
- If it guesses right, say yes. Otherwise, pick from the list.  
- Enter your image path (e.g., `photos/my_dog.jpg`).  
- It processes, saves as `output_my_dog.jpg`, and opens it automatically.  
### Step 5: Testing  
The test file is included in the project which you can run using pytest  

### If Stuff Goes Wrong  
Cut the genie some slack because its his first day at work (And also mine 😅)  
  
## Conclusion  
  
It's not perfect (the text matching isn't Google-level smart), but it helped me learn about image processing and various python concepts like ascii and making TUI's  
  