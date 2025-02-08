+++
title = "Exploring the Gopher Protocol"
description = "Documenting Gopher protocol."
categories =  [
	"web-protocols"
]
date = "2021-06-29"
author = "Anirudh Ganesh"
[taxonomies]
tags = [
    "gopher-protocol",
    "web-protocols",
    "retro-internet",
    "minimal-web",
    "tech-history",
    "lightweight-browsing",
    "text-based-internet",
    "privacy-focused-web",
    "alternative-internet",
    "open-source",
    "networking",
    "command-line-browsing",
    "low-bandwidth-solutions",
    "protocol-comparison",
    "information-retrieval"
]

[extra]
toc = true
katex = true
+++

The Gopher protocol, often overshadowed by the rise of the World Wide Web,
remains an intriguing and efficient way to access and share information online.
Originally developed at the University of Minnesota in the early 1990s, Gopher provided a hierarchical, text-based browsing experience that emphasized simplicity and speed.
Despite its decline in mainstream usage, the protocol continues to have a dedicated following and offers valuable insights into lightweight, structured content distribution.

## Understanding the Gopher Protocol

Gopher is a client-server protocol designed for distributing, searching, and retrieving documents over the Internet.
Unlike the web, which allows for complex hypertext linking and multimedia content, Gopher structures information into a series of menus and files, presented in a line-based format.

### Core Features
When I first stumbled upon Gopher, what immediately stood out was how organized and hierarchical it was.
Unlike the modern web, which often feels like an endless labyrinth of links, Gopher structures information in a way that just makes sense.
Everything is neatly arranged into menus and documents, making it easy to navigate without the distractions of ads, pop-ups, or flashy visuals.
It reminded me of the early days of exploring file systems—simple, direct, and efficient.

Another thing I found fascinating was how lightweight it is.
Because Gopher doesn’t rely on heavy multimedia or bloated scripts, it consumes barely any bandwidth.
It’s the kind of system that could still work beautifully on dial-up or in places with spotty internet.
One particular place where this would've been cool to see in the early days of Kindle which came with unlimited
free 3G internet, when Amazon started limiting access to only Amazon ebooks and Wikipedia, I believe if Kindles
came with Gopher support, it would've been cool to have explored the Gopherspace on a kindle.
It's one of those devices which is also very text focused and it would've been great to see.

Gopher also takes a text-first approach, focusing purely on content without unnecessary formatting.
There’s something refreshing about that—no autoplay videos, no massive images breaking layouts, just straightforward information.
It’s the kind of simplicity I sometimes miss when trying to read an article online and having to wade through cookie banners, newsletters, and a dozen other distractions.

What’s even cooler is how versatile its linking system is.
Gopher doesn’t just connect documents and directories; it can link to all kinds of external services, even Telnet.
This opens up possibilities beyond simple browsing, making it feel more like a lightweight, interconnected internet.
I can imagine how there could be an alternate universe where Gopher took off and people watch youtube through telnet ascii art videos, like that one
Star Wars ascii art video on telnet.

Gopher might be a relic of an earlier internet era, but its design choices—efficiency, simplicity, and structure—still have a lot to offer.
In a world where the web is growing heavier and more complex, revisiting something so streamlined is a refreshing reminder that sometimes, less is more.
I wonder how an interactive service would look like in the Gopherspace, something like the equivalent of today's SPA in React/Vue etc.

## Gopher Protocol Specification

Digging into the technical details of the Gopher protocol to try and understand how it works is pretty interesting.
It seems the protocol is designed to save bandwidth and keep things simple,
I guess this is because it was designed in the early days of the internet when they might not have had the bandwidth we have today.
The Gopher protocol is defined by RFC 1436 and consists of a simple request-response mechanism:

* **Client Request**: The client sends a single-line request to the server, containing a selector string that identifies the resource.
* **Server Response**: The server returns either a directory listing (menu) or the requested file content.
* **Data Format**: Each line in a directory listing follows this structure:

```
typeIndicator <TAB> displayName <TAB> selector <TAB> hostname <TAB> port
```

### Gopher Item Types

Each resource in a Gopher menu is identified by a single-character type code:

| Type Code | Description |
| ---- | ---- |
| 0 | Plain Text File |
| 1 | Directory (Menu) |
| 7 | Full-text Search Query |
| h | HTML File (Non-standard) |
| g | GIF Image (Non-standard) |
| I | Binary File/Image |

## Detailed Technical Aspects

Gopher operates over TCP/IP and listens on port 70, as assigned by IANA. When a client connects, it sends a selector string, which can be empty, to the server. The server processes this request and responds with either:

A directory listing containing item types, user-visible names, selector strings, hostnames, and port numbers.

A document consisting of a simple text block terminated by a period on a line by itself.

For a full-text search query (type 7), the client sends a query string in addition to the selector, and the server responds with a virtual directory containing matching entries.

The protocol's simplicity allows debugging via telnet by manually sending requests and observing responses.

### Developing for Gopher

Developing Gopher content involves structuring a gophermap file that defines how clients interpret and present your data. A typical Gopher menu might look like this:

```
0Welcome to My Gopher Space<TAB>/welcome.txt<TAB>gopher.example.com<TAB>70
1About<TAB>/about<TAB>gopher.example.com<TAB>70
7Search<TAB>/search<TAB>gopher.example.com<TAB>70
```

### Gopher Search Engines

Several search services index Gopherspace and provide full-text search capabilities:

* **Floodgap** (gopher://gopher.floodgap.com): A well-maintained Gopher search and directory service.
* **Gopherpedia**: A Gopher version of Wikipedia.

## Conclusion

Exploring the Gopher protocol has been a fascinating journey into the early days of the internet.
Reading about it has motivated me to set up my own blog on Gopherspace as a fun side project.
If I have the free time, I might even try to build a Gopher server/creating some kind of analogue to the static site generators we have today.
