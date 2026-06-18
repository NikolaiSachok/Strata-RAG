<?php
  // FICTIONAL sample contact page — mirrors the real corpus's simple email template so the
  // BOUNDED contact-email derivation (harvest.derive_contact_emails) can be tested. The page
  // builds a support address as <literal-local-part>@<?= ...domain... ?>. The PHP domain token
  // is NEVER executed by the harvester — it only recognises the LITERAL local-part 'support'
  // and joins it to the domain harvested from config.yaml. All content here is invented.
  $config = require __DIR__ . '/config.php';
?>
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Vista Weather — Contact</title></head>
<body>
  <h1>Contact us</h1>
  <p>Questions about Vista Weather? Email us at
     support@<?= $config['app']['domain'] ?> and we'll reply within two business days.</p>
</body>
</html>
