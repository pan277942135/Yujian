# Chromium display fixtures

The Chromium CI job downloads these public fixture photos, verifies their source SHA-256, and creates resized derivatives under the ignored `test-results/` directory. The photos are used only by the DET_FISH display smoke test.

- The landscape source is [Fish caught with a rod in Gambia](https://commons.wikimedia.org/wiki/File:Fish_caught_with_a_rod_in_Gambia.jpg), by Peter van der Sluijs, licensed under CC BY-SA 3.0. Source SHA-256: `6117d964dd10ee8f3f38d82b78441c7e03a2402cbfd623e6d05d9d2d75685581`.
- The portrait source is [Portrait of man on boat holding a caught fish (AM 81857-1)](https://commons.wikimedia.org/wiki/File:Portrait_of_man_on_boat_holding_a_caught_fish_(AM_81857-1).jpg), Auckland Museum, licensed under CC BY 4.0. Source SHA-256: `d806491e4711a58aa242c8ef3feb6fcc0e9b13c56c5db5e7808203134b3c1ffa`.
- The near-square case is a proportional, padded square derivative of the landscape photo above and retains the full source photo.

The browser harness uses a clearly labelled simulated API response to check image rendering only. It does not claim to run the production detector.
