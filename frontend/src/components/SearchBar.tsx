import { useState } from "react";

interface Props {
  onSearch: (query: string) => void;
  disabled?: boolean;
}

const EXAMPLES = [
  "a slow-burn sci-fi like Arrival but funnier",
  "feel-good 90s comedies",
  "mind-bending thrillers with a twist",
  "cozy animated movies for a rainy day",
];

export default function SearchBar({ onSearch, disabled }: Props) {
  const [value, setValue] = useState("");

  function submit(query: string) {
    const q = query.trim();
    if (q) onSearch(q);
  }

  return (
    <div>
      <form
        className="searchbar"
        onSubmit={(e) => {
          e.preventDefault();
          submit(value);
        }}
      >
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Describe a vibe…  e.g. a slow-burn sci-fi like Arrival but funnier"
          aria-label="Describe a vibe"
        />
        <button className="btn" type="submit" disabled={disabled}>
          Search
        </button>
      </form>
      <div className="chips">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            className="chip"
            onClick={() => {
              setValue(ex);
              submit(ex);
            }}
          >
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}
