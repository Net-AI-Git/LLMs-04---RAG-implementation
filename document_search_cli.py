#!/usr/bin/env python3
"""
Document Search System - CLI Interface
A command-line interface for indexing and searching documents using RAG technology.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List

from index_documents import process_document
from search_documents import search_query
from utils.logging_utils import configure_logging, get_logger
from utils.db_utils import get_indexed_filenames, delete_document_data, DatabaseError

# Configure logging at the very start, based on command-line arguments.
# This must be done before any other module calls get_logger().
parser = argparse.ArgumentParser(
    description="A command-line interface for indexing and searching documents."
)
parser.add_argument(
    '--debug',
    action='store_true',
    help="Enable debug mode to show detailed internal logs on the console."
)
args = parser.parse_args()
configure_logging(debug_mode=args.debug)

# Now it's safe to get the logger.
logger = get_logger(__name__)


def print_banner():
    """Display the application banner."""
    print("\n" + "=" * 50)
    print("📚 DOCUMENT SEARCH SYSTEM")
    print("   Powered by RAG Technology")
    print("=" * 50)


def print_main_menu():
    """Display the main menu options."""
    print("\n📋 MAIN MENU")
    print("-" * 20)
    print("1. 📄 Add New Document")
    print("2. 🔍 Search Documents")
    print("3. 🗄️ Manage Database")
    print("4. 🚪 Exit")
    print()


def get_user_choice(max_option: int, prompt: str = "Choose an option") -> int:
    """
    Get and validate user's menu choice.

    Args:
        max_option (int): Maximum valid option number
        prompt (str): The prompt to display to the user.

    Returns:
        int: User's choice (1 to max_option)
    """
    while True:
        try:
            choice_str = input(f"{prompt} (1-{max_option}): ").strip()
            if not choice_str:
                continue
            choice_num = int(choice_str)

            if 1 <= choice_num <= max_option:
                return choice_num
            else:
                print(f"❌ Please enter a number between 1 and {max_option}")

        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)


def list_files_in_directory(directory: str = ".") -> List[Path]:
    """
    List PDF and DOCX files in the specified directory.

    Args:
        directory (str): Directory path to scan

    Returns:
        List[Path]: List of Path objects for supported document files
    """
    try:
        path = Path(directory)
        if not path.is_dir():
            return []

        supported_extensions = ['.pdf', '.docx']
        files = [
            file_path for file_path in path.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions
        ]
        return sorted(files)

    except Exception as e:
        logger.error(f"Error listing files in '{directory}': {e}")
        return []


def _process_single_file(file_path: Path) -> None:
    """
    Helper function to process a single file and print status messages.

    Args:
        file_path (Path): The path to the file to be processed.
    """
    print(f"\n🔄 Processing document: {file_path.name}")
    print("   This may take a few moments...")
    try:
        success = process_document(str(file_path))
        if success:
            print(f"✅ Document '{file_path.name}' processed and indexed successfully!")
        else:
            print(f"❌ Failed to process document '{file_path.name}'. Check logs for details if in debug mode.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing '{file_path}': {e}")
        print(f"❌ An unexpected error occurred with '{file_path.name}'.")


def _prompt_for_folder_path(prompt_message: str) -> Optional[Path]:
    """
    Prompts the user for a folder path and validates it.

    Args:
        prompt_message (str): The message to display to the user.

    Returns:
        Optional[Path]: A valid Path object to a directory, or None if input is invalid or cancelled.
    """
    folder_path_str = input(f"\n{prompt_message}: ").strip()
    if not folder_path_str:
        print("❌ No path entered. Returning to menu.")
        return None

    folder_path = Path(folder_path_str)
    if not folder_path.is_dir():
        print(f"❌ Error: The path '{folder_path}' is not a valid folder.")
        return None

    return folder_path


def _handle_browse_from_cwd() -> None:
    """
    Handles the workflow for browsing from the current working directory.
    """
    current_dir = Path.cwd()
    while True:
        print(f"\n🧭 Current Directory: {current_dir}")
        print("-" * 40)

        items = sorted(list(current_dir.iterdir()), key=lambda p: (not p.is_dir(), p.name.lower()))
        directories = [p for p in items if p.is_dir()]
        files = [p for p in items if p.is_file() and p.suffix.lower() in ['.pdf', '.docx']]

        options = []
        if current_dir.parent != current_dir:
            options.append(("parent", "📁 .. (Parent Directory)"))
        options.extend([("dir", d) for d in directories])
        options.extend([("file", f) for f in files])
        options.append(("back", "🔙 Back to Add Document Menu"))

        for i, (item_type, item_path) in enumerate(options, 1):
            if item_type == "file":
                print(f"{i}. 📄 {item_path.name}")
            elif item_type == "dir":
                print(f"{i}. 📁 {item_path.name}/")
            else:
                print(f"{i}. {item_path}")

        if not directories and not files:
            print("   (No supported documents or folders found)")
        print()

        choice = get_user_choice(len(options), "Select an item or action")
        selected_type, selected_item = options[choice - 1]

        if selected_type == "back":
            return
        elif selected_type == "parent":
            current_dir = current_dir.parent
        elif selected_type == "dir":
            current_dir = selected_item
        elif selected_type == "file":
            _process_single_file(selected_item)
            return


def _handle_select_file_from_folder() -> None:
    """
    Handles the workflow for selecting a single file from a user-provided folder path.
    """
    try:
        folder_path = _prompt_for_folder_path("Please paste the path to the folder")
        if not folder_path:
            return

        files = list_files_in_directory(str(folder_path))
        if not files:
            print(f"📂 No supported documents (.pdf, .docx) found in '{folder_path}'.")
            return

        print("\nPlease select a file to process:")
        for i, file_path in enumerate(files, 1):
            print(f"{i}. 📄 {file_path.name}")
        print(f"{len(files) + 1}. 🔙 Cancel")

        choice = get_user_choice(len(files) + 1, "Select a file")

        if choice == len(files) + 1:
            print("❌ Operation cancelled.")
            return

        selected_file = files[choice - 1]
        _process_single_file(selected_file)

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled.")
    except Exception as e:
        logger.error(f"An error occurred while selecting file from folder: {e}")
        print("❌ An unexpected error occurred.")


def _handle_process_folder() -> None:
    """
    Handles the workflow for processing all supported files within a user-provided folder path.
    """
    try:
        folder_path = _prompt_for_folder_path("Please paste the path to the folder to process")
        if not folder_path:
            return

        files_to_process = list_files_in_directory(str(folder_path))
        if not files_to_process:
            print(f"📂 No supported documents (.pdf, .docx) found in '{folder_path}'.")
            return

        print(f"\nFound {len(files_to_process)} supported documents. Starting processing...")
        for i, file_path in enumerate(files_to_process, 1):
            print("-" * 40)
            print(f"Processing file {i} of {len(files_to_process)}...")
            _process_single_file(file_path)

        print("\n" + "=" * 40)
        print(f"✅ Folder processing complete. Processed {len(files_to_process)} files.")
        print("=" * 40)

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled.")
    except Exception as e:
        logger.error(f"An error occurred while processing a folder: {e}")
        print("❌ An unexpected error occurred.")


def add_document_menu():
    """Handle the add document workflow."""
    print("\n📄 ADD NEW DOCUMENT")
    print("-" * 25)
    print("1. 🧭 Browse from current directory")
    print("2. 📂 Select file from a specific folder")
    print("3. 🗂️ Process an entire folder")
    print("4. 🔙 Back to main menu")
    print()

    choice = get_user_choice(4)

    if choice == 1:
        _handle_browse_from_cwd()
    elif choice == 2:
        _handle_select_file_from_folder()
    elif choice == 3:
        _handle_process_folder()
    elif choice == 4:
        return


def search_documents_menu():
    """Handle the search documents workflow."""
    print("\n🔍 SEARCH DOCUMENTS")
    print("-" * 20)
    print("Enter your search query below:")
    print("(Press Enter with empty query to return to main menu)")
    print()

    try:
        query = input("🔎 Search: ").strip()

        if not query:
            print("❌ Search cancelled")
            return

        print(f"\n🔄 Searching for: '{query}'")
        print("   Please wait...")

        # Perform the search
        results = search_query(query)

        # Display results
        print("\n" + "=" * 60)
        print(results)
        print("=" * 60)

        # Wait for user to read results
        input("\nPress Enter to continue...")

    except KeyboardInterrupt:
        print("\n❌ Search cancelled")
    except Exception as e:
        logger.error(f"Error during search: {e}")
        # Display a user-friendly message instead of the raw error.
        print("❌ An unexpected error occurred during search. For details, run with --debug.")


def _handle_list_documents():
    """Handles listing all indexed documents."""
    print("\nℹ️ Listing all indexed documents...")
    try:
        filenames = get_indexed_filenames()
        if not filenames:
            print("   The database is currently empty.")
            return

        print("-" * 40)
        for filename in filenames:
            print(f"  📄 {Path(filename).name}")
        print("-" * 40)

    except DatabaseError as e:
        print(f"❌ A database error occurred: {e}")


def _handle_delete_specific_document():
    """Handles deleting a specific document from the database."""
    print("\n🗑️ Select a document to delete:")
    try:
        filenames = get_indexed_filenames()
        if not filenames:
            print("   The database is currently empty. Nothing to delete.")
            return

        for i, filename in enumerate(filenames, 1):
            print(f"  {i}. 📄 {Path(filename).name}")
        print(f"  {len(filenames) + 1}. 🔙 Cancel")

        choice = get_user_choice(len(filenames) + 1, "Select a document to delete")
        if choice == len(filenames) + 1:
            print("❌ Deletion cancelled.")
            return

        file_to_delete = filenames[choice - 1]
        print(f"\n⚠️ Are you sure you want to delete all data for '{Path(file_to_delete).name}'?")
        confirmation = input("   Type 'YES' to confirm: ").strip()

        if confirmation == 'YES':
            if delete_document_data(filename=file_to_delete):
                print(f"✅ Successfully deleted '{Path(file_to_delete).name}'.")
            else:
                print(f"❌ Failed to delete '{Path(file_to_delete).name}'.")
        else:
            print("❌ Deletion cancelled.")

    except (DatabaseError, KeyboardInterrupt) as e:
        print(f"❌ An error occurred: {e}")


def _handle_delete_all_documents():
    """Handles clearing the entire documents database."""
    print("\n🔥🔥🔥 WARNING: THIS WILL DELETE ALL INDEXED DATA. 🔥🔥🔥")
    print("This action is irreversible.")
    confirmation = input("To proceed, type 'DELETE ALL' and press Enter: ").strip()

    if confirmation == 'DELETE ALL':
        print("\n🔄 Deleting all data...")
        if delete_document_data():
            print("✅ Database has been cleared successfully.")
        else:
            print("❌ Failed to clear the database.")
    else:
        print("❌ Deletion cancelled. No data was changed.")


def manage_database_menu():
    """Display the database management menu and handle user choices."""
    print("\n🗄️ MANAGE DATABASE")
    print("-" * 25)
    print("1. ℹ️ List Indexed Documents")
    print("2. 🗑️ Delete a Specific Document")
    print("3. 🔥 Delete ALL Documents (Clear Database)")
    print("4. 🔙 Back to Main Menu")
    print()

    choice = get_user_choice(4)

    if choice == 1:
        _handle_list_documents()
    elif choice == 2:
        _handle_delete_specific_document()
    elif choice == 3:
        _handle_delete_all_documents()
    elif choice == 4:
        return


def main():
    """Main CLI application loop."""
    try:
        print_banner()

        while True:
            print_main_menu()
            choice = get_user_choice(4)

            if choice == 1:
                add_document_menu()
            elif choice == 2:
                search_documents_menu()
            elif choice == 3:
                manage_database_menu()
            elif choice == 4:
                print("\n👋 Thank you for using Document Search System!")
                print("   Goodbye!")
                break

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main loop: {e}")
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        print("   For more details, run with the --debug flag.")


if __name__ == "__main__":
    main()
