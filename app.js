async function sendMessage() {
    let userInput = document.getElementById("user-input");
    let message = userInput.value.trim();
    if (!message) return;

    // Show user message
    let chatBox = document.getElementById("chat-box");
    let userHtml = `<div class="user-message"><b>You:</b> ${message}</div>`;
    chatBox.innerHTML += userHtml;

    // Send to Flask backend
    const response = await fetch("/get", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message })
    });

    const data = await response.json();

    // Show bot response
    let botHtml = `<div class="bot-message"><b>Bot:</b> ${data.response}</div>`;
    chatBox.innerHTML += botHtml;

    // Scroll to bottom
    chatBox.scrollTop = chatBox.scrollHeight;

    // Clear input
    userInput.value = "";
}

// ✅ Allow "Enter" key to send message
document.getElementById("user-input").addEventListener("keypress", function(e) {
    if (e.key === "Enter") {
        sendMessage();
    }
});
