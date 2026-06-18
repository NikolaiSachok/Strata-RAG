<?php
  // Server-side config — must NEVER reach the index. Tests the markup stripper drops it.
  $api_key = "should_be_stripped_with_the_php_block";
  $title = "Vista Weather";
?>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Vista Weather — beautiful forecasts</title>
  <style>body { font-family: sans-serif; } .hero { color: navy; }</style>
  <script>console.log("analytics that should be stripped");</script>
</head>
<body>
  <section class="hero">
    <h1>Vista Weather</h1>
    <p>Vista Weather is a calm, mountain-themed weather app. Each forecast is shown as a
       layered ridgeline that shifts colour with the conditions &mdash; misty blues for rain,
       warm amber for clear skies.</p>
    <p>Plan your week with an hourly mountain panorama, severe-weather alerts, and a gentle
       sunrise-to-sunset gradient. No mascot, no jokes &mdash; just a serene alpine view.</p>
  </section>
</body>
</html>
