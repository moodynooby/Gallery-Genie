from art import *
from ascii_magic import AsciiArt, from_image
def main():
    print(text2art("Welcome to Gallery Genie (G.G) Inc","cybermedium "))
    print('Umm.... Hello I am a intern at G.G , and this is my first day so ... I am sorry in advance')
    my_art = AsciiArt.from_image('Genie.jpg')
    print(my_art.to_ascii(width_ratio=3.7,columns=60))
    name = input(' I am G whats your name: ')

    print(text2art('Hello '+name,'rounded'))

if __name__ == "__main__":
    main()