import pygame

from ui import ArcadeUI


def main() -> None:
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("Move On - Demo")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    app = ArcadeUI(screen=screen, config_path="config.json")
    app.run()
    pygame.quit()


if __name__ == "__main__":
    main()
