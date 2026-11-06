# Using the Demo

This page explains what the concierge can do and how to ask for it. The demo runs at
**[iellis02-blue-horizon.hf.space](https://iellis02-blue-horizon.hf.space)**. Sign-in is
required, with either a Hugging Face account (preferred) or a Google account. Google
sign-in may show an unverified app warning, and it is limited to 100 users in total, ever;
once those are used, only Hugging Face sign-in works.

## Demo video

<video controls preload="metadata" style="width: 100%;">
  <source src="https://media.githubusercontent.com/media/iannellis/Blue-Horizon-AI-Concierge/main/demo-video.webm" type="video/webm">
  Your browser cannot play this video.
  <a href="https://media.githubusercontent.com/media/iannellis/Blue-Horizon-AI-Concierge/main/demo-video.webm">Download it instead</a>.
</video>

## What to expect

Once signed in, you browse as one of the hotel's fictional guests, with that guest's
existing reservations in the sidebar. The concierge does two jobs: it answers questions
about the hotel, and it searches for and books rooms. It politely refuses anything else.

- **The room calendar covers January 4, 2025 to January 3, 2026.** The hotel has no rooms
  outside that range. If you give a date without a year, the concierge picks the one
  that falls inside it.
- **Bookings are real writes to the demo database,** which is reset every night, so book,
  modify, and cancel freely.
- **Nothing is booked until you click Confirm.** When you ask to book, cancel, or change a
  reservation, the concierge prepares the change and the app shows a confirmation dialog
  with the exact details. The confirmation number appears only after you confirm.
- **Limits keep the public demo affordable:** 10 messages per rolling hour, and 60
  room-nights per guest in any 24 hours (one room for one night is one room-night).
- **Each search shows a few rooms,** four by default. Ask for more and it shows up to 15,
  along with how many rooms matched in total.

## Booking a room

**Rooms are booked by room number, not by room type.** A normal hotel sells you "a king
room" and assigns the actual room at check-in. Here, every one of the hotel's 600 rooms
is its own bookable item. The concierge searches for rooms that fit your request and
shows you specific rooms, such as room 1204, each with its own floor, view, size,
amenities, and price. You then book the room you want by its number, for example *book
room 1204* or *book the second one*. One booking can cover several rooms.

You can also ask to see your reservations, cancel one, or change its dates or rooms.

### What you can search for

Any of these can be combined in one request, with or without dates.

| You can ask about | Values |
|---|---|
| Dates | A check-in and check-out date. Only rooms free for every night come back, with the stay's total price. Without dates, the search covers rooms in general. |
| Room type | Standard, Deluxe, Suite, Presidential Suite |
| Bed type | King, Queen, Double King, Double Queen, King + Sofa Bed, King + Multiple Sofa Beds |
| View | Ocean View, Panoramic Ocean View, City View, Courtyard View, Pool View, Corner View, Standard View |
| Floor | Floors 1 to 20, as a minimum, a maximum, or a range ("the top floor" works too) |
| Guests | How many guests the room sleeps, from 2 to 6, as a minimum or a maximum |
| Size | Square feet, from 350 to 2,999, as a minimum or a maximum |
| Accessibility | Accessible rooms only, or rooms that are not accessible |
| Nightly price | A minimum or maximum price per night; needs dates |
| Specific rooms | One or more room numbers |
| Amenities | The room must have all the amenities you name (see below) |

Broad requests reach the premium variants too: "a suite" includes Presidential Suites,
and "an ocean view" includes Panoramic Ocean View.

Results can be sorted by stay price, floor, room size, or room number, either lowest or
highest first. Sorting by price needs dates.

Room amenities you can require include a balcony or wraparound balcony, private terrace,
private pool, Jacuzzi tub, soaking tub, sauna, steam room or steam shower, full kitchen
or kitchenette, dining area or formal dining room, living room, walk-in closet, work desk
or executive office, wine fridge, grand piano, private gym equipment, lounge or executive
lounge access, butler service, a private chef, luxury car service, a Nespresso machine,
and premium sound systems from Bose or Bang & Olufsen.

## Asking about the hotel

The information agent answers from three sources: frequently asked questions, the
hotel's amenities, and its spa and fitness services.

**Frequently asked questions** cover booking, amenities, services, and hotel policies,
such as check-in and check-out times or late check-out.

**Amenities** span these categories:

| Category | Examples |
|---|---|
| Spa | Swedish, deep tissue, and hot stone massage; couples massage; facials; manicure and pedicure; aromatherapy |
| Fitness | Personal training, yoga, Pilates, group and aqua fitness classes, meditation, beach fitness |
| Room service | Breakfast in bed, evening dining, late-night menu, romantic dinner setup, dietary specialist menu, celebration package |
| Entertainment | Private movie screening, live music, magic show |
| Business | Private conference room, executive assistant, business center access |
| Childcare | Kids club, private babysitting, children's activity program |
| Transportation | Airport transfer, city tour, VIP shopping transfer |
| Water sports | Private yacht charter, scuba diving |

**Services** are bookable spa and fitness sessions of set lengths: spa treatments,
massage therapy, personal training, wellness consultations, fitness assessments, and
group fitness classes.

### Restrictions you can add

Amenity and service questions can carry these constraints, and the concierge only shows
what satisfies them:

| Restriction | Example |
|---|---|
| Price | *a massage under $150*, *something between $50 and $100* |
| Duration | *a spa treatment of at least 90 minutes*, *a class under an hour* |
| Notice | *I can only give 4 hours' notice* |
| Booking | *something I can walk into without a reservation* |

If nothing matches, the concierge says so and suggests what is available nearby rather
than inventing a service. The concierge describes these amenities and services but does
not book them; only rooms can be booked through the chat.
