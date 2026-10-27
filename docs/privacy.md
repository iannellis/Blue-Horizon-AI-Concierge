# Privacy Policy

*Last updated: October 27, 2026*

Blue Horizon Concierge is a portfolio project by Ian Ellis. The hotel is fictional, and
no booking made in the demo is real. This page describes what the live demo at
[iellis02-blue-horizon.hf.space](https://iellis02-blue-horizon.hf.space) does with
information about you.

## Information collected

### Your Google account

You sign in with Google so the demo can limit how many messages each visitor sends per
hour. Signing in shares your name, email address, and profile picture with the demo.
The demo uses them for two things only:

- showing who is signed in, in the sidebar
- counting your messages against the hourly limit

Your name and email address are held in the demo's memory and in a sign-in cookie in
your browser. They are not written to a database or to any log, and the hourly count
is cleared whenever the demo restarts, which happens at least once a day.

### Your messages

What you type is sent to [OpenAI](https://openai.com/policies/privacy-policy/) to
produce the concierge's replies. Under OpenAI's API terms, it is not used to train
their models, and OpenAI may keep it for up to 30 days to monitor for abuse.

The conversation is also kept in the demo's memory so the concierge can follow it, and
is cleared whenever the demo restarts. Conversations may be recorded in
[LangSmith](https://www.langchain.com/privacy-policy), a debugging service, so that
errors can be traced and the concierge improved.

Please do not enter real personal, financial, or health information. The demo never
needs it.

### Bookings

Each visitor is assigned a fictional guest at random. Bookings you make are stored
under that guest in the demo's database, which is reset to its original state every
night.

### Operational logs

The demo logs events such as request timings, errors, and model usage to
[Axiom](https://axiom.co/privacy), a logging service. These logs identify a
conversation only by a random ID and the fictional guest's number. They never contain
your name or email address. They do not record your messages or the concierge's
replies, but they do record the room-search filters a request used (such as dates and
room type), and a report of an error can include part of what the concierge derived
from your message.

## What is not done with it

Your information is not sold, shared for advertising, or used to contact you. No
analytics or advertising trackers run on the demo. The only cookie it sets is the
sign-in cookie.

The demo is hosted on [Hugging Face Spaces](https://huggingface.co/privacy), which
handles your connection to it under its own privacy policy.

## Your choices

You can remove the demo's access to your Google account at any time from
[your Google account's permissions page](https://myaccount.google.com/permissions).

Use of information received from Google APIs adheres to the
[Google API Services User Data Policy](https://developers.google.com/terms/api-services-user-data-policy),
including its Limited Use requirements.

## Contact

Questions about this policy can be raised as an issue on the project's
[GitHub repository](https://github.com/iannellis/Blue-Horizon-AI-Concierge/issues).

Use of the demo is also covered by the [terms of service](terms.md).

## Changes

If this policy changes, the date at the top of this page will change with it.
