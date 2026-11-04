// Setup page: shows drive/Code.gs (published as drive-script.txt) with a copy button.
(async () => {
  const code = document.getElementById('code');
  const msg = document.getElementById('copyMsg');
  try {
    code.value = await (await fetch('drive-script.txt')).text();
  } catch {
    msg.textContent = 'کد بارگیری نشد؛ اینترنت را بررسی کنید.';
  }
  document.getElementById('copyBtn').onclick = async () => {
    try {
      await navigator.clipboard.writeText(code.value);
      msg.textContent = 'کپی شد.';
    } catch {
      code.select(); // older browsers: let the user copy the selection
      msg.textContent = 'متن انتخاب شد؛ آن را کپی کنید.';
    }
  };
})();
