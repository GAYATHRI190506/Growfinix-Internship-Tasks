def generate_tags(objects, colors, lighting):

    tags = []

    tags.extend(objects)

    for color in colors:
        tags.append(f"RGB{tuple(color)}")

    tags.append(lighting)

    return tags