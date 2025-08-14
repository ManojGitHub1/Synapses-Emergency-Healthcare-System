// This file is ready for any future JavaScript interactivity.
console.log("Emergency Response System JS loaded.");


console.log("Emergency Response System JS loaded.");

// Utility: gzip JSON and return base64 string
function gzipJSON(obj) {
    const bytes = new TextEncoder().encode(JSON.stringify(obj));
    // Browser GZIP compression
    const gz = fflate.gzipSync(bytes);
    return btoa(String.fromCharCode(...gz)); // convert to base64 for safe transmission
}

// Example: send compressed report
async function sendCompressedReport(data) {
    const compressed = gzipJSON(data);
    return fetch("/report", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-Content-Encoded": "gzip-b64"
        },
        body: JSON.stringify({ data: compressed })
    });
}
