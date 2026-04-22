def mock_lead_capture(name: str, email: str, platform: str) -> str:
    print(f"\n✅ Lead captured successfully: {name}, {email}, {platform}\n")
    return f"Lead captured successfully for {name} ({email}) on {platform}."